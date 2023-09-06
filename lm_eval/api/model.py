import abc
import os

from typing import Union, List, Tuple
from sqlitedict import SqliteDict
import json
import hashlib

from tqdm import tqdm

from lm_eval import utils
from lm_eval.logger import eval_logger


class LM(abc.ABC):
    def __init__(self) -> None:
        """Defines the interface that should be implemented by all LM subclasses.
        LMs are assumed to take text (strings) as input and yield strings as output
        (inputs/outputs should be tokenization-agnostic.)

        """
        # set rank and world size to a single process, by default.
        self._rank = 0
        self._world_size = 1
        self.cache_hook = CacheHook(None)

    @abc.abstractmethod
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.

        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.

        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """
        pass

    @abc.abstractmethod
    def loglikelihood_rolling(self, requests) -> List[Tuple[float, bool]]:
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: EOT
            Max context length: 4
            Resulting input/prediction pairs:

                INPUT:  EOT   0   1   2
                PRED:     0   1   2   3

                INPUT:    3   4   5   6
                PRED:     4   5   6   7

                INPUT:    5   6   7   8
                PRED:             8   9

          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, continuation).
            string: str
                String for which we are computing per-token loglikelihood
        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            logprob: float
                The log probability of `continuation`
            isgreedy:
                Whether `continuation` would be generated by greedy sampling from `context`
        """
        pass

    # TODO: Add an optional max length
    @abc.abstractmethod
    def greedy_until(self, requests) -> List[str]:
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, until).
            context: str
                Context string
            until: [str]
                The string sequences to generate until. These string sequences
                may each span across multiple tokens, or may be part of one token.
        :return: list[str]
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        pass

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        if args2.get("device") == "mps" or args.get("device") == "mps":
            args["dtype"] = "float32"
        return cls(**args, **args2)

    @property
    def rank(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._rank

    @property
    def world_size(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._world_size

    def set_cache_hook(self, cache_hook) -> None:
        self.cache_hook = cache_hook


### SQLite-based caching of LM responses
def hash_args(attr, args):
    dat = json.dumps([attr] + list(args))
    return hashlib.sha256(dat.encode("utf-8")).hexdigest()


class CacheHook:
    def __init__(self, cachinglm) -> None:
        if cachinglm is None:
            self.dbdict = None
            return

        self.dbdict = cachinglm.dbdict

    def add_partial(self, attr, req, res) -> None:
        if self.dbdict is None:
            return
        hsh = hash_args(attr, req)
        self.dbdict[hsh] = res


class CachingLM:
    def __init__(self, lm, cache_db) -> None:
        """LM wrapper that returns cached results if they exist, and uses the underlying LM if not.

        :param lm: LM
            Underlying LM
        :param cache_db: str
            Path to cache db
        """
        self.lm = lm
        self.cache_db = cache_db
        if os.path.dirname(cache_db):
            os.makedirs(os.path.dirname(cache_db), exist_ok=True)
        self.dbdict = SqliteDict(cache_db, autocommit=True)

        # add hook to lm
        lm.set_cache_hook(self.get_cache_hook())

    def __getattr__(self, attr):
        lm_attr = getattr(self.lm, attr)
        if not callable(lm_attr):
            return lm_attr

        def fn(requests):
            res = []
            remaining_reqs = []
            warned = False
            # figure out which ones are cached and which ones are new
            eval_logger.info(
                f"Loading '{attr}' responses from cache '{self.cache_db}' where possible..."
            )
            for req in tqdm(requests):
                hsh = hash_args(attr, req.args)
                if attr == "greedy_until" and req.args[1].get("do_sample", False):
                    # when we are doing non-greedy generation, don't use the cache
                    # (else every "randomly sampled" generation would be identical for repeats > 1).
                    if not warned:
                        eval_logger.warning(
                            f"Arguments to lm.greedy_until() '{req.args[1]}' include non-deterministic sampling. Caching will not be performed for such requests."
                        )
                        warned = True
                    res.append(None)
                    remaining_reqs.append(req)
                elif hsh in self.dbdict:
                    ob = self.dbdict[hsh]

                    assert ob is not None

                    res.append(ob)
                else:
                    res.append(None)
                    remaining_reqs.append(req)

            # actually run the LM on the requests that do not have cached results
            rem_res = getattr(self.lm, attr)(remaining_reqs)

            # stick the new ones back into the list and also cache any of the new ones
            resptr = 0
            for req, r in zip(remaining_reqs, rem_res):
                while res[resptr] is not None:
                    resptr += 1

                res[resptr] = r

                # caching
                hsh = hash_args(attr, req.args)
                self.dbdict[hsh] = r
            self.dbdict.commit()

            return res

        return fn

    def get_cache_hook(self):
        return CacheHook(self)
