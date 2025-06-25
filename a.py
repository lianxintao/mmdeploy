@torch.no_grad()
def event_loop_pp_disagg_prefill(self: Scheduler) -> None:
    """A pipeline parallel scheduler loop for prefill worker in disaggregation mode."""
    
    # Initialize microbatch structures for pipeline parallelism
    mbs = [None] * self.pp_size
    last_mbs = [None] * self.pp_size
    self.running_mbs = [
        ScheduleBatch(reqs=[], batch_is_full=False) for _ in range(self.pp_size)
    ]
    bids = [None] * self.pp_size
    pp_outputs: Optional[PPProxyTensors] = None
    
    while True:
        server_is_idle = True
        
        for mb_id in range(self.pp_size):
            self.running_batch = self.running_mbs[mb_id]
            self.last_batch = last_mbs[mb_id]

            # Process incoming requests - similar to disagg_prefill
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            
            # Handle bootstrapped requests from disagg queue
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            
            # Process prefill chunks for current microbatch
            self.process_prefill_chunk()
            
            # Get new batch for prefill processing
            mbs[mb_id] = self.get_new_batch_prefill()
            
            # Handle MLP sync if required
            if require_mlp_sync(self.server_args):
                mbs[mb_id], _ = self.prepare_mlp_sync_batch(mbs[mb_id])
                
            self.running_mbs[mb_id] = self.running_batch
            self.cur_batch = mbs[mb_id]
            
            if self.cur_batch:
                server_is_idle = False
                result = self.run_batch(self.cur_batch)

                # (last rank) send the outputs to the next step
                if self.pp_group.is_last_rank:
                    next_token_ids, bids[mb_id] = (
                        result.next_token_ids,
                        result.bid,
                    )
                    if self.cur_batch.return_logprob:
                        pp_outputs = PPProxyTensors(
                            {
                                "next_token_ids": next_token_ids,
                                "extend_input_len_per_req": result.extend_input_len_per_req,
                                "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
                            }
                            | (
                                {
                                    f"logits_output.{k}": v
                                    for k, v in result.logits_output.__dict__.items()
                                }
                                if result.logits_output is not None
                                else {}
                            )
                        )
                    else:
                        pp_outputs = PPProxyTensors(
                            {
                                "next_token_ids": next_token_ids,
                            }
                        )
                    # send the output from the last round to let the next stage worker run post processing
                    self.pp_group.send_tensor_dict(
                        pp_outputs.tensors,
                        all_gather_group=self.attn_tp_group,
                    )

            # receive outputs and post-process (filter finished reqs) the coming microbatch
            next_mb_id = (mb_id + 1) % self.pp_size
            next_pp_outputs = None
            if mbs[next_mb_id] is not None:
                next_pp_outputs: Optional[PPProxyTensors] = PPProxyTensors(
                    self.pp_group.recv_tensor_dict(
                        all_gather_group=self.attn_tp_group
                    )
                )
                mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]
                logits_output_args = {
                    k[len("logits_output.") :]: v
                    for k, v in next_pp_outputs.tensors.items()
                    if k.startswith("logits_output.")
                }
                if len(logits_output_args) > 0:
                    logits_output = LogitsProcessorOutput(**logits_output_args)
                else:
                    logits_output = None
                output_result = GenerationBatchResult(
                    logits_output=logits_output,
                    pp_hidden_states_proxy_tensors=None,
                    next_token_ids=next_pp_outputs["next_token_ids"],
                    extend_input_len_per_req=next_pp_outputs.tensors.get(
                        "extend_input_len_per_req", None
                    ),
                    extend_logprob_start_len_per_req=next_pp_outputs.tensors.get(
                        "extend_logprob_start_len_per_req", None
                    ),
                    bid=bids[next_mb_id],
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                # Use disagg_prefill specific batch result processing
                self.process_batch_result_disagg_prefill(mbs[next_mb_id], output_result)
                last_mbs[next_mb_id] = mbs[next_mb_id]

            # (not last rank)
            if not self.pp_group.is_last_rank:
                if self.cur_batch:
                    bids[mb_id] = result.bid
                # carry the outputs to the next stage
                # send the outputs from the last round to let the next stage worker run post processing
                if pp_outputs:
                    self.pp_group.send_tensor_dict(
                        pp_outputs.tensors,
                        all_gather_group=self.attn_tp_group,
                    )

                # send out reqs to the next stage
                dp_offset = self.attn_dp_rank * self.attn_tp_size
                if self.attn_tp_rank == 0:
                    point_to_point_pyobj(
                        recv_reqs,
                        self.pp_rank * self.tp_size + dp_offset,
                        self.world_group.cpu_group,
                        self.pp_rank * self.tp_size + dp_offset,
                        (self.pp_rank + 1) * self.tp_size + dp_offset,
                    )

                # send out proxy tensors to the next stage
                if self.cur_batch:
                    self.pp_group.send_tensor_dict(
                        result.pp_hidden_states_proxy_tensors,
                        all_gather_group=self.attn_tp_group,
                    )

            pp_outputs = next_pp_outputs
            
            # Process inflight disagg prefill queue for current microbatch
            if len(self.disagg_prefill_inflight_queue) > 0:
                self.process_disagg_prefill_inflight_queue()
            
            # Reset batch_is_full flag for each microbatch to prevent hanging under high concurrency
            # This is critical for PD separation mode
            self.running_mbs[mb_id].batch_is_full = False

        # When the server is idle, self-check and re-init some states
        # Only do this when all microbatches are None and inflight queue is empty
        all_mbs_none = all(mb is None for mb in mbs)
        if server_is_idle and all_mbs_none and len(self.disagg_prefill_inflight_queue) == 0:
            self.check_memory()
            self.new_token_ratio = self.init_new_token_ratio
            self.maybe_sleep_on_idle()
