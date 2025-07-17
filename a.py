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


from sglang.srt.model_executor.logits_processor import LogitsProcessorOutput
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import require_mlp_sync, point_to_point_pyobj
from sglang.srt.managers.scheduler import PPProxyTensors



   @torch.no_grad()
    def event_loop_pp_disagg_decode(self: Scheduler):
        """A pipeline parallelism scheduler loop for decode worker in disaggregation mode."""
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

                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                # polling and allocating kv cache
                self.process_decode_queue()
                mbs[mb_id] = self.get_next_disagg_decode_batch_to_run()
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch = mbs[mb_id]
                prepare_mlp_sync_flag = require_mlp_sync(self.server_args)

                if self.cur_batch:
                    server_is_idle = False
                    # Generate fake extend output.
                    if self.cur_batch.forward_mode.is_extend():
                        # Note: Logprobs should be handled on the prefill engine.
                        self.stream_output(
                            self.cur_batch.reqs, any(req.return_logprob for req in self.cur_batch.reqs)
                        )
                        if prepare_mlp_sync_flag:
                            self._prepare_idle_batch_and_run(None)
                    else:
                        if prepare_mlp_sync_flag:
                            self.prepare_mlp_sync_batch(self.cur_batch)
                        result = self.run_batch(self.cur_batch)

                # (last rank) send the outputs to the next step
                if self.pp_group.is_last_rank:
                    if self.cur_batch and not self.cur_batch.forward_mode.is_extend():
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
                    self.process_batch_result(mbs[next_mb_id], output_result)
                    last_mbs[next_mb_id] = mbs[next_mb_id]

                # (not last rank)
                if not self.pp_group.is_last_rank:
                    if self.cur_batch and not self.cur_batch.forward_mode.is_extend():
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
                    if self.cur_batch and not self.cur_batch.forward_mode.is_extend():
                        self.pp_group.send_tensor_dict(
                            result.pp_hidden_states_proxy_tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            if server_is_idle and (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
                == 0
            ):
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()
现在的disa
        logger.info(
            f"decode rank: engine_rank={self.kv_mgr.kv_args.engine_rank}, "
            f"target_tp_ranks={self.target_tp_ranks}, target_dp_group={self.target_dp_group}, "
            f"bootstrap_addr={self.bootstrap_addr}"
        )
ggregation 不支持 pipline 并行，原因是在prefill，decode 的文件中并未像scheduler文件中那样单独实现了适应与P/D disaggregation脚骨的 event_loop_pp函数，并排查其他相关问题，给我增加PD分离模式的pipline 并行功能，保宁完善



        # PP stage之间同一个req的kv_receiver状态同步
        if hasattr(self.scheduler, 'pp_group') and self.scheduler.pp_group.world_size > 1:
            self._sync_kv_receiver_states_across_pp_stages(polls)

    def _sync_kv_receiver_states_across_pp_stages(self, polls):
        """
        在PP stage之间同步kv_receiver状态，使用all_reduce MIN操作确保状态一致性
        类似于 dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=pp_group)
        """
        try:
            current_rank = self.scheduler.pp_group.rank_in_group
            pp_world_size = self.scheduler.pp_group.world_size
            
            if pp_world_size <= 1:
                return  # 只有一个PP stage，无需同步
            
            # 构建request ID到本地队列索引的映射
            req_id_to_local_idx = {}
            for i, decode_req in enumerate(self.queue):
                req_id_to_local_idx[decode_req.req.rid] = i
            
            # 收集所有PP stages的kv_receiver状态
            all_stages_states = self._collect_all_pp_stages_states(polls)
            
            # 对于每个请求，使用MIN操作同步状态
            sync_updates = self._apply_min_reduce_logic(all_stages_states, req_id_to_local_idx, polls)
            
            # 输出同步日志
            if sync_updates:
                logger.info(
                    f"PP Stage {current_rank} kv_receiver sync updates:\n" +
                    "\n".join(sync_updates)
                )
            else:
                logger.debug(f"PP Stage {current_rank} kv_receiver states already consistent")
            
            # 最终barrier同步，确保所有stages完成状态同步
            self.scheduler.pp_group.barrier()
            
        except Exception as e:
            logger.warning(f"PP kv_receiver state sync failed on stage {current_rank}: {e}")
            # 同步失败时继续执行，避免阻塞整个流程
            pass

    def _collect_all_pp_stages_states(self, polls):
        """收集所有PP stages的kv_receiver状态"""
        current_rank = self.scheduler.pp_group.rank_in_group
        pp_world_size = self.scheduler.pp_group.world_size
        
        # 构建当前stage的状态
        current_stage_requests = {}
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            current_stage_requests[decode_req.req.rid] = {
                "poll_status": int(poll),
                "bootstrap_room": decode_req.req.bootstrap_room,
                "local_idx": i
            }
        
        # 收集所有stages的状态
        all_stages_states = [None] * pp_world_size
        all_stages_states[current_rank] = current_stage_requests
        
        # 与其他stages交换状态信息
        for other_rank in range(pp_world_size):
            if other_rank == current_rank:
                continue
                
            # 使用rank顺序避免死锁
            if current_rank < other_rank:
                self.scheduler.pp_group.send_object(current_stage_requests, dst=other_rank)
                other_stage_requests = self.scheduler.pp_group.recv_object(src=other_rank)
            else:
                other_stage_requests = self.scheduler.pp_group.recv_object(src=other_rank)
                self.scheduler.pp_group.send_object(current_stage_requests, dst=other_rank)
            
            all_stages_states[other_rank] = other_stage_requests
        
        return all_stages_states

    def _apply_min_reduce_logic(self, all_stages_states, req_id_to_local_idx, polls):
        """
        应用MIN reduce逻辑，对于相同的kv_receiver取最小状态值
        类似于: tensor_to_reduce = min(tensor_to_reduce) across all PP stages
        """
        current_rank = self.scheduler.pp_group.rank_in_group
        sync_updates = []
        
        # 收集所有stages中出现的请求ID
        all_req_ids = set()
        for stage_requests in all_stages_states:
            if stage_requests is not None:
                all_req_ids.update(stage_requests.keys())
        
        # 对每个请求ID进行MIN reduce操作
        for req_id in all_req_ids:
            if req_id not in req_id_to_local_idx:
                continue  # 当前stage没有这个请求，跳过
                
            local_idx = req_id_to_local_idx[req_id]
            current_status = int(polls[local_idx])
            
            # 收集所有stages中这个请求的状态
            req_statuses = []
            stages_with_req = []
            
            for stage_rank, stage_requests in enumerate(all_stages_states):
                if stage_requests is not None and req_id in stage_requests:
                    req_statuses.append(stage_requests[req_id]["poll_status"])
                    stages_with_req.append(stage_rank)
            
            if len(req_statuses) > 1:
                # 使用MIN操作找到最小状态值（最保守的状态）
                min_status = min(req_statuses)
                
                if min_status != current_status:
                    # 更新本地状态为最小值
                    polls[local_idx] = min_status
                    sync_updates.append(
                        f"Req {req_id}: MIN reduce updated poll_status from {current_status} "
                        f"to {min_status} (stages: {stages_with_req}, statuses: {req_statuses})"
                    )
                else:
                    logger.debug(
                        f"Req {req_id}: Already at MIN status {current_status} "
                        f"(stages: {stages_with_req}, statuses: {req_statuses})"
                    )
        
        return sync_updates



# DeepSeek 32B 多机LWS配置
# 需要至少2台8卡机器

---
# Prefill Service
apiVersion: v1
kind: Service
metadata:
  name: deepseek32b-prefill-service
spec:
  selector:
    leaderworkerset.sigs.k8s.io/name: deepseek32b-main
    role: leader
    environment: test
    release: test
  ports:
    - protocol: TCP
      port: 30000
      targetPort: 30000

---
# Decode Service
apiVersion: v1
kind: Service
metadata:
  name: deepseek32b-decode-service
spec:
  selector:
    leaderworkerset.sigs.k8s.io/name: deepseek32b-main
    role: worker
    environment: test
    release: test
    environment: test
    release: test
  ports:
    - protocol: TCP
      port: 30001
      targetPort: 30001

---
# Load Balancer Service
apiVersion: v1
kind: Service
metadata:
  name: deepseek32b-lb-service
spec:
  type: NodePort
  selector:
    leaderworkerset.sigs.k8s.io/name: deepseek32b-main
    role: worker
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
      nodePort: 30800

---
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: deepseek32b-main
  labels:
    yice: "true"
    environment: test
    release: test
spec:
  leaderWorkerTemplate:
    # Leader Template - Prefill 节点（使用一台完整的8卡机器）
    leaderTemplate:
      metadata:
        labels:
          role: leader
          component: prefill
          yice: "true"
          environment: test
          release: test
      spec:
        containers:
        - name: sglang-prefill
          image: lmsysorg/sglang:latest
          command:
          - python3
          - -m
          - sglang.launch_server
          - --port
          - "30000"
          - --host
          - "0.0.0.0"
          - --model-path
          - /work/models
          - --chunked-prefill-size
          - "262144"
          - --max-prefill-tokens
          - "16384"
          - --page-size
          - "64"
          - --enable-dp-attention
          - --enable-dp-lm-head
          - --dp-size
          - "2"
          - --enable-deepep-moe
          - --deepep-mode
          - normal
          - --disaggregation-mode
          - prefill
          - --mem-fraction-static
          - "0.7"
          - --context-length
          - "16384"
          - --tp-size
          - "8"  # 使用全部8张GPU
          - --disaggregation-ib-device
          - mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
          - --trust-remote-code
          - --disaggregation-ib-device
          - mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
          - --ep-num-redundant-experts
          - "16"
          - --moe-dense-tp-size
          - "1"
          - --max-running-requests
          - "512"
          env:
          - name: CUDA_VISIBLE_DEVICES
            value: "0,1,2,3,4,5,6,7"
          - name: NVSHMEM_HCA_PE_MAPPING
            value: "mlx5_bond_0:1:2,mlx5_bond_1:1:2,mlx5_bond_2:1:2,mlx5_bond_3:1:2"
          - name: NVSHMEM_IB_GID_INDEX
            value: "3"
          - name: NVSHMEM_ENABLE_NIC_PE_MAPPING
            value: "1"
          - name: SGLANG_SET_CPU_AFFINITY
            value: "true"
          - name: SGL_ENABLE_JIT_DEEPGEMM
            value: "1"
          - name: NCCL_IB_QPS_PER_CONNECTION
            value: "8"
          - name: NCCL_IB_SPLIT_DATA_ON_QPS
            value: "1"
          - name: NCCL_NET_PLUGIN
            value: none
          - name: NCCL_IB_TC
            value: "136"
          - name: NCCL_MIN_NCHANNELS
            value: "4"
          - name: MC_TE_METRIC
            value: "false"
          - name: NCCL_IB_SL
            value: "5"
          - name: NCCL_IB_HCA
            value: ^=mlx5_0,mlx5_5,mlx5_6
          ports:
          - containerPort: 30000
            protocol: TCP
          readinessProbe:
            periodSeconds: 30
            tcpSocket:
              port: 30000
          resources:
            limits:
              nvidia.com/gpu: "8"  # 使用全部8张GPU
          securityContext:
            capabilities:
              add:
              - IPC_LOCK
            privileged: true
          volumeMounts:
          - mountPath: /dev/shm
            name: dshm
          - mountPath: /work/models
            name: model
          - mountPath: /dev/infiniband
            name: ib
          - mountPath: /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs
            name: cf
          - mountPath: /root/.cache
            name: sgl-cache
        
        # 使用 affinity 选择匹配的机器
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
              - matchExpressions:
                - key: pd
                  operator: In
                  values:
                  - "yes"
                - key: gpu-type
                  operator: In
                  values:
                  - "L20"
                - key: infiniband
                  operator: In
                  values:
                  - "enabled"
            preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              preference:
                matchExpressions:
                - key: node-type
                  operator: In
                  values:
                  - "prefill"
        tolerations:
        - key: bopd
          operator: Exists
        - key: node-role
          operator: Exists
        volumes:
        - emptyDir:
            medium: Memory
          name: dshm
        - hostPath:
            path: /data1/maas_hosted_models/models/DeepSeek-32B
          name: model
        - hostPath:
            path: /dev/infiniband
          name: ib
        - hostPath:
            path: /data1/maas_hosted_models/models/fused_moe_triton/configs
          name: cf
        - hostPath:
            path: /data1/sgl_cache
            type: DirectoryOrCreate
          name: sgl-cache
    
    # Worker Template - Decode节点（使用另一台完整的8卡机器）
    workerTemplate:
      metadata:
        labels:
          role: worker
          component: decode-lb
          yice: "true"
          environment: test
          release: test
      spec:
        containers:
        # Decode容器
        - name: sglang-decode
          image: lmsysorg/sglang:latest
          command:
          - python3
          - -m
          - sglang.launch_server
          - --port
          - "30001"
          - --host
          - "0.0.0.0"
          - --model-path
          - /work/models
          - --chunked-prefill-size
          - "262144"
          - --page-size
          - "64"
          - --enable-dp-attention
          - --enable-dp-lm-head
          - --dp-size
          - "2"
          - --enable-deepep-moe
          - --deepep-mode
          - low_latency
          - --disaggregation-mode
          - decode
          - --mem-fraction-static
          - "0.8"
          - --context-length
          - "16384"
          - --cuda-graph-max-bs
          - "64"
          - --max-running-requests
          - "1024"
          - --tp-size
          - "8"  # 使用全部8张GPU
          - --trust-remote-code
          - --ep-num-redundant-experts
          - "16"
          - --moe-dense-tp-size
          - "1"
          env:
          - name: CUDA_VISIBLE_DEVICES
            value: "0,1,2,3,4,5,6,7"
          - name: NVSHMEM_IB_TRAFFIC_CLASS
            value: "16"
          - name: NVSHMEM_IB_GID_INDEX
            value: "3"
          - name: NVSHMEM_ENABLE_NIC_PE_MAPPING
            value: "1"
          - name: NVSHMEM_HCA_PE_MAPPING
            value: "mlx5_bond_0:1:2,mlx5_bond_1:1:2,mlx5_bond_2:1:2,mlx5_bond_3:1:2"
          - name: NCCL_IB_QPS_PER_CONNECTION
            value: "8"
          - name: NCCL_IB_SPLIT_DATA_ON_QPS
            value: "1"
          - name: NCCL_NET_PLUGIN
            value: "none"
          - name: NCCL_IB_TC
            value: "136"
          - name: NCCL_MIN_NCHANNELS
            value: "4"
          - name: MC_TE_METRIC
            value: "true"
          - name: NCCL_IB_SL
            value: "5"
          - name: SGLANG_MOONCAKE_TRANS_THREAD
            value: "16"
          - name: SGL_ENABLE_JIT_DEEPGEMM
            value: "1"
          - name: NCCL_IB_HCA
            value: ^=mlx5_0,mlx5_5,mlx5_6
          ports:
          - containerPort: 30001
            protocol: TCP
          readinessProbe:
            periodSeconds: 30
            tcpSocket:
              port: 30001
          resources:
            limits:
              nvidia.com/gpu: "8"  # 使用全部8张GPU
          securityContext:
            capabilities:
              add:
              - IPC_LOCK
            privileged: true
          volumeMounts:
          - mountPath: /root/.cache
            name: sgl-cache
          - mountPath: /dev/shm
            name: dshm
          - mountPath: /work/models
            name: model
          - mountPath: /dev/infiniband
            name: ib
          - mountPath: /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs
            name: cf
        
        # Load Balancer容器
        - name: sgl-loadbalancer
          image: lmsysorg/sglang:latest
          command:
          - python
          - -m
          - sglang.srt.disaggregation.mini_lb
          - --prefill
          - http://deepseek32b-prefill-service:30000
          - --decode
          - http://localhost:30001
          - --host
          - 0.0.0.0
          - --port
          - "8000"
          ports:
          - containerPort: 8000
            protocol: TCP
          readinessProbe:
            periodSeconds: 30
            tcpSocket:
              port: 8000
          resources:
            limits:
              cpu: "2"
              memory: "4Gi"
        
        # 使用 affinity 选择匹配的机器，确保调度到不同的节点
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
              - matchExpressions:
                - key: pd
                  operator: In
                  values:
                  - "yes"
                - key: gpu-type
                  operator: In
                  values:
                  - "L20"
                - key: infiniband
                  operator: In
                  values:
                  - "enabled"
            preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              preference:
                matchExpressions:
                - key: node-type
                  operator: In
                  values:
                  - "decode"
          podAntiAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchExpressions:
                - key: role
                  operator: In
                  values:
                  - leader
                - key: component
                  operator: In
                  values:
                  - prefill
              topologyKey: kubernetes.io/hostname
        tolerations:
        - key: bopd
          operator: Exists
        - key: node-role
          operator: Exists
        volumes:
        - hostPath:
            path: /data1/sgl_cache1
            type: DirectoryOrCreate
          name: sgl-cache
        - emptyDir:
            medium: Memory
          name: dshm
        - hostPath:
            path: /dev/infiniband
          name: ib
        - hostPath:
            path: /data1/maas_hosted_models/models/DeepSeek-32B
          name: model
        - hostPath:
            path: /data1/maas_hosted_models/models/fused_moe_triton/configs
          name: cf
    
    restartPolicy: RecreateGroupOnPodRestart
    size: 1
  
  networkConfig:
    subdomainPolicy: Shared
  replicas: 1
  rolloutStrategy:
    rollingUpdateConfiguration:
      maxSurge: 0
      maxUnavailable: 1
    type: RollingUpdate
  startupPolicy: LeaderCreated
