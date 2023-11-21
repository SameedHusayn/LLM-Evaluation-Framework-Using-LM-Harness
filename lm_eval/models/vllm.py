from collections import defaultdict
from typing import List, Tuple, Optional, Literal

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
import copy
from tqdm import tqdm
from lm_eval.api.registry import register_model
from lm_eval import utils
from vllm import LLM, SamplingParams


@register_model("vllm")
class VLLM(LM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained="gpt2",
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_gen_toks: int = 256,
        swap_space: int = 4,
        batch_size: int = 1,
        max_length: int = None,
    ):
        super().__init__()

        self.model = LLM(
            model=pretrained,
            revision=revision,
            dtype=dtype,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            swap_space=swap_space,
            quantization=quantization,
        )
        self.tokenizer = self.model.get_tokenizer()
        self.batch_size = batch_size
        self._max_length = max_length
        self._max_gen_toks = max_gen_toks

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if hasattr(self.model.llm_engine.model_config, "max_model_len"):
            return self.model.llm_engine.model_config.max_model_len
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=False):
        """ """
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def _model_generate(
        self,
        requests: List[int] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        if "do_sample" in kwargs.keys():
            kwargs.pop("do_sample")
        if generate:
            generate_sampling_params = SamplingParams(
                max_tokens=max_tokens, stop=stop, **kwargs
            )
            outputs = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=generate_sampling_params,
            )
        else:
            logliklihood_sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=2, max_tokens=1
            )
            outputs = self.model.generate(
                prompt_token_ids=requests, sampling_params=logliklihood_sampling_params
            )
        return outputs

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = [
                    self.eot_token_id
                ], self.tokenizer.tok_encode(continuation)
            else:
                context_enc, continuation_enc = self.tokenizer(
                    [context, continuation],
                    truncation="do_not_truncate",
                    add_special_tokens=False,
                    return_attention_mask=False,
                ).input_ids

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests) -> List[float]:
        loglikelihoods = []

        for (string,) in tqdm([req.args for req in requests]):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = defaultdict(list)
        re_ords = {}

        # batch tokenize contexts
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding = self.tokenizer(context)
        requests = list(zip((context, context_encoding.input_ids), all_gen_kwargs))

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), tuple(_requests[0][1])

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        grouper = utils.Grouper(requests, lambda x: str(x[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = utils.Reorderer(requests, _collate_gen)

        pbar = tqdm(total=len(requests), disable=(self.rank != 0))
        # for each different set of kwargs, we execute all requests, by batch.
        for key, re_ord in re_ords.items():
            chunks = utils.chunks(
                re_ord.get_reordered(),
                n=self.batch_size,
                fn=None,
            )
            for chunk in chunks:
                context_and_encoding, all_gen_kwargs = zip(*chunk)
                context, context_encoding = context_and_encoding
                # we assume all gen kwargs in the batch are the same
                # this is safe to assume because the `grouper` object ensures it.
                gen_kwargs = all_gen_kwargs[0]
                # unpack our keyword arguments.
                until = None
                if isinstance(gen_kwargs, dict):
                    kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [until]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                            )
                else:
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}"
                    )
                if not until:
                    until = [self.tokenizer.decode(self.eot_token_id)]
                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.max_gen_toks

                # set the max length in tokens of inputs ("context_enc")
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                context_encoding = [x[-max_ctx_len:] for x in context_encoding]

                # TODO: max_length in kwargs

                # perform batched generation
                cont = self._model_generate(
                    requests=context_encoding,
                    generate=True,
                    max_tokens=max_gen_toks,
                    stop=until,
                    **kwargs,
                )

                # cache generations
                for output, context in zip(cont, context):
                    generated_text = output.outputs[0].text
                    res[key].append(generated_text)
                    self.cache_hook.add_partial(
                        "generate_until", (context, gen_kwargs), generated_text
                    )
                    pbar.update(1)

            # reorder this group of results back to original unsorted form
            res[key] = re_ord.get_original(res[key])

        pbar.close()

        return grouper.get_original(res)

    def _loglikelihood_tokens(
        self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        chunks = utils.chunks(
            re_ord.get_reordered(),
            n=self.batch_size,
            fn=None,
        )
        pbar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)))
        for chunk in chunks:
            inps = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inps.append(inp)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inps, generate=False)

            for output, ctxlen, (cache_key, context_enc, continuation_enc) in zip(
                outputs, ctxlens, chunk
            ):
                answer = self._parse_logprobs(
                    (context_enc + continuation_enc),
                    output,
                    ctxlen,
                )

                res.append(answer)

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                    pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens: List, outputs, ctxlen: int):
        """Process logprobs and tokens.

        :param tokens: list
            Tokens from response
        :param outputs: list
            Logprobabilities tied to tokens
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """

        # Calculate continuation_logprobs
        continuation_logprobs = sum(
            logprob_dict[token]
            for token, logprob_dict in zip(
                tokens[ctxlen:], outputs.prompt_logprobs[ctxlen:]
            )
        )

        # Determine if is_greedy
        is_greedy = True
        for i in range(ctxlen, len(tokens)):
            token = tokens[i]
            top_tokens = outputs[i]
            top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
            if top_token != token:
                is_greedy = False
                break

        return continuation_logprobs, is_greedy
