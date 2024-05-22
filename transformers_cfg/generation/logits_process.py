import copy
import math
import os
import pprint

import torch
import logging
from line_profiler import profile
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
from transformers.utils import add_start_docstrings

logger = logging.getLogger(__name__)


class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        grammar_constraint,
        parse_start_index=None,
    ):
        self.last_size = None
        self.grammar_constraint = grammar_constraint
        self.batch_accept_states = None
        self.prev_accept_states = None
        self.parse_start_index = None

    def undo_last_step(self):
        if self.prev_accept_states is None:
            raise ValueError("No previous accept states to revert to")
        self.batch_accept_states = self.prev_accept_states
        self.prev_accept_states = None
        self.grammar_constraint.undo_last_step()

    def mask_logits(self, logits, device, batch_idx=None):
        masked_logits = logits.clone()
        # resolve each stack to a tensor of True/False for each token
        # indicating acceptance
        # acceptance = self.grammar_acceptor.filter_vocab(self.stacks, device)
        if batch_idx is not None:
            acceptance = self.grammar_constraint.batch_filter_vocab(
                [self.batch_accept_states[batch_idx]], device
            )
        else:
            acceptance = self.grammar_constraint.batch_filter_vocab(
                self.batch_accept_states, device
            )
        # acceptance is a tensor of shape (batch_size, vocab_size)
        # get the indices of the accepted tokens
        # do the following operation only in debug mode
        if os.getenv("DEBUG_MODE") == "True":
            # convert acceptance to numpy array
            batch_size, vocab_size = acceptance.shape
            acceptance_np = acceptance.cpu().numpy()
            accepted_x, accepted_y = acceptance_np.nonzero()
            # dict of {batch_index: [accepted_token_indices]}
            # initialize the dict with empty list
            accepted_token_indices = {i: [] for i in range(batch_size)}
            for x, y in zip(accepted_x, accepted_y):
                accepted_token_indices[x].append(y)
            logger.debug("Accepted token indices for the current batch:")
            logger.debug("\n" + pprint.pformat(accepted_token_indices))
            # convert token_ids to tokens
            accepted_tokens = {
                i: [
                    self.grammar_constraint.tokenizer.decode([token_id])
                    for token_id in token_ids
                ]
                for i, token_ids in accepted_token_indices.items()
            }
            logger.debug("Accepted tokens for the current batch:")
            logger.debug("\n" + pprint.pformat(accepted_tokens))

        # Logits to -inf where False
        masked_logits[~acceptance] = -math.inf
        return masked_logits

    # TODO: batching
    def process_logits(self, input_ids, scores, batch_idx=None):
        """
        :param input_ids:
        :param scores:
        :return:
        """
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_accept_states is None:
            self.batch_accept_states = [
                # self.grammar_constraint.init_stacks()
                copy.deepcopy(
                    self.grammar_constraint.string_recognizer.get_initial_accept_state()
                )
                for _ in range(len(input_ids))
            ]

        if os.getenv("DEBUG_MODE") == "True":
            print("-" * 80)

        logger.debug("input_ids: \n" + pprint.pformat(input_ids))
        # logger.debug("scores: \n" + pprint.pformat(scores))
        logger.debug("last_size: \n" + pprint.pformat(self.last_size))
        logger.debug(
            "num of stacks: \n"
            + pprint.pformat(
                [len(acc_state.stacks) for acc_state in self.batch_accept_states]
            )
        )
        # logger.debug("stacks: \n" + pprint.pformat(self.batch_accept_states.stacks))
        if batch_idx is not None:
            self.batch_accept_states[batch_idx] = (
                self.grammar_constraint.consume_token_ids(
                    input_ids,
                    [self.batch_accept_states[batch_idx]],
                    self.parse_start_index,
                )[0]
            )

        else:
            self.batch_accept_states = self.grammar_constraint.consume_token_ids(
                input_ids,
                self.batch_accept_states,
                self.parse_start_index,
            )
        self.prev_accept_states = copy.deepcopy(self.batch_accept_states)
        logger.debug(f"input_ids: {input_ids}")

        masked_scores = self.mask_logits(scores, scores.device, batch_idx)
        return masked_scores

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    @profile
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        batch_idx: int = None,
    ) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores, batch_idx)
