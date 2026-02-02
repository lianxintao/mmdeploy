https://github.com/NVIDIA/TransformerEngine
https://github.com/NVIDIA/TileGym/tree/main/src/tilegym/ops/cutile
https://github.com/NVIDIA/cutile-python
https://github.com/NVIDIA/accelerated-computing-hub/blob/main/Accelerated_Python_User_Guide/notebooks/Chapter_01_GPU_Computing_Basics.ipynb
https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
https://zhuanlan.zhihu.com/p/639297098
https://zhuanlan.zhihu.com/p/5750410146
https://zhuanlan.zhihu.com/p/518857175
https://developer.download.nvidia.cn/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf
https://zhuanlan.zhihu.com/p/555339335
https://github.com/xlite-dev/HGEMM
https://zhuanlan.zhihu.com/p/555339335
https://zhuanlan.zhihu.com/p/669926191
https://zhuanlan.zhihu.com/p/4496065391
https://zhuanlan.zhihu.com/p/441146275
https://zhuanlan.zhihu.com/p/584236348
https://github.com/AyakaGEMM/Hands-on-GEMM/tree/main/src/cuda
https://github.com/ifromeast/cuda_learning/blob/main/03_gemm/sgemm_v1.cu
https://github.com/AyakaGEMM/Hands-on-GEMM/blob/main/src/cuda/i8tc_ptx_cutlass_k32_gemm.cu
https://zhuanlan.zhihu.com/p/703256080


42.194.203.117:21116
42.194.203.117:21117
+Fa1jUWPsB2QxKC4+Ivk51ZswxliJjtWNTTPZY7J0tU=

110.40.200.222:21116
110.40.200.222:21117
ancFyyI87muqvtIvO8rJziPkOJSWs1Dq1kTroXtTFqo=
https://triton-lang.cn/main/getting-started/tutorials/index.html

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

# DeepSeek 32B 多机部署配置
# 需要至少2台8卡机器
# 使用 hostNetwork 直接通过宿主机IP访问，无需Service

---
# Prefill Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek32b-prefill
  labels:
    app: deepseek32b-prefill
    yice: "true"
    environment: test
    release: test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: deepseek32b-prefill
      role: prefill
  template:
    metadata:
      labels:
        app: deepseek32b-prefill
        role: prefill
        component: prefill
        yice: "true"
        environment: test
        release: test
    spec:
      containers:
      - name: sglang-prefill
        image: aicr.byd.com/docker.io/lmsysorg/sglang:v0.4.7-cu124-post1
        command:
        - python3
        - -m
        - sglang.launch_server
        - --port
        - "30000"
        - --host
        - "0.0.0.0"
        - --model-path
        - /models/DeepSeek-R1-Distill-Qwen-32B
        - --page-size
        - "64"
        - --disaggregation-mode
        - prefill
        - --mem-fraction-static
        - "0.85"
        - --tp-size
        - "8"  # 使用全部8张GPU
        - --disaggregation-ib-device
        - mlx5_bond_0
        - --trust-remote-code
        - --quantization
        - fp8
        - --kv-cache-dtype
        - fp8_e5m2
        - --attention-backend
        - flashinfer
        env:
        - name: NVSHMEM_HCA_PE_MAPPING
          value: "mlx5_bond_0:1:2"
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
          value: "mlx5_bond_0"
        - name: NCCL_SOCKET_IFNAME
          value: "bond1"
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
        - mountPath: /models
          name: host-models
        - mountPath: /dev/infiniband
          name: ib

      # 使用 affinity 选择匹配的机器
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: org
                operator: In
                values:
                - "yiceai"
              - key: yiceai
                operator: In
                values:
                - "true"
              - key: deploy
                operator: In
                values:
                - "deepseekr1-32b-pd-p"
        # 反亲和性确保 prefill 和 decode 不在同一节点
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - deepseek32b-decode
            topologyKey: kubernetes.io/hostname
      volumes:
      - emptyDir:
          medium: Memory
        name: dshm
      - hostPath:
          path: /export/models
        name: host-models
      - hostPath:
          path: /dev/infiniband
        name: ib
      dnsPolicy: Default
      hostIPC: true
      hostNetwork: true  # 使用宿主机网络

---
# Decode Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek32b-decode
  labels:
    app: deepseek32b-decode
    yice: "true"
    environment: test
    release: test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: deepseek32b-decode
      role: decode
  template:
    metadata:
      labels:
        app: deepseek32b-decode
        role: decode
        component: decode
        yice: "true"
        environment: test
        release: test
    spec:
      containers:
      - name: sglang-decode
        image: aicr.byd.com/docker.io/lmsysorg/sglang:v0.4.7-cu124-post1
        command:
        - python3
        - -m
        - sglang.launch_server
        - --port
        - "30001"
        - --host
        - "0.0.0.0"
        - --model-path
        - /models/DeepSeek-R1-Distill-Qwen-32B
        - --page-size
        - "64"
        - --disaggregation-mode
        - decode
        - --mem-fraction-static
        - "0.85"
        - --tp-size
        - "8"  # 使用全部8张GPU
        - --disaggregation-ib-device
        - mlx5_bond_0
        - --trust-remote-code
        - --quantization
        - fp8
        - --kv-cache-dtype
        - fp8_e5m2
        - --attention-backend
        - flashinfer
        env:
        - name: NVSHMEM_HCA_PE_MAPPING
          value: "mlx5_bond_0:1:2"
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
          value: "mlx5_bond_0"
        - name: NCCL_SOCKET_IFNAME
          value: "bond1"
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
        - mountPath: /dev/shm
          name: dshm
        - mountPath: /models
          name: host-models
        - mountPath: /dev/infiniband
          name: ib

      # 使用 affinity 选择匹配的机器
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: org
                operator: In
                values:
                - "yiceai"
              - key: yiceai
                operator: In
                values:
                - "true"
              - key: deploy
                operator: In
                values:
                - "deepseekr1-32b-pd-p"
        # 反亲和性确保 prefill 和 decode 不在同一节点
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - deepseek32b-prefill
            topologyKey: kubernetes.io/hostname
      volumes:
      - emptyDir:
          medium: Memory
        name: dshm
      - hostPath:
          path: /export/models
        name: host-models
      - hostPath:
          path: /dev/infiniband
        name: ib
      dnsPolicy: Default
      hostIPC: true
      hostNetwork: true  # 使用宿主机网络

---
# Load Balancer Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek32b-loadbalancer
  labels:
    app: deepseek32b-loadbalancer
    yice: "true"
    environment: test
    release: test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: deepseek32b-loadbalancer
      role: loadbalancer
  template:
    metadata:
      labels:
        app: deepseek32b-loadbalancer
        role: loadbalancer
        component: loadbalancer
        yice: "true"
        environment: test
        release: test
    spec:
      containers:
      - name: sgl-loadbalancer
        image: lmsysorg/sglang:latest
        command:
        - python
        - -m
        - sglang.srt.disaggregation.mini_lb
        - --prefill
        - $(PREFILL_HOST_URL)
        - --decode
        - $(DECODE_HOST_URL)
        - --host
        - 0.0.0.0
        - --port
        - "8000"
        env:
        - name: PREFILL_HOST_URL
          value: "http://192.168.1.100:30000"  # 替换为实际的 prefill 宿主机IP
        - name: DECODE_HOST_URL
          value: "http://192.168.1.101:30001"  # 替换为实际的 decode 宿主机IP
        readinessProbe:
          periodSeconds: 30
          tcpSocket:
            port: 8000
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"

      # 使用 affinity 选择匹配的机器，建议与 prefill/decode 分离
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: org
                operator: In
                values:
                - "yiceai"
              - key: yiceai
                operator: In
                values:
                - "true"
              - key: deploy
                operator: In
                values:
                - "deepseekr1-32b-pd-p"
        # 反亲和性确保 LB 不与 prefill/decode 在同一节点（可选）
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - deepseek32b-prefill
                  - deepseek32b-decode
              topologyKey: kubernetes.io/hostname
      hostNetwork: true  # 使用宿主机网络
      dnsPolicy: Default


#!/bin/bash

set -e  # 遇到错误时退出脚本

# 安装 git 和 git-lfs（适配 Debian/Ubuntu，其他系统需手动替换）
if ! command -v git &> /dev/null; then
    echo "安装 git..."
    sudo apt-get update
    sudo apt-get install -y git
else
    echo "git 已安装"
fi

if ! command -v git-lfs &> /dev/null; then
    echo "安装 git-lfs..."
    sudo apt-get install -y git-lfs
else
    echo "git-lfs 已安装"
fi

# 激活 git-lfs
echo "激活 git-lfs..."
git lfs install

# 创建并进入目标目录
mkdir -p ms-models
cd ms-models

# Clone 模型仓库
echo "Cloning Qwen3-235B-A22B-Instruct-2507-FP8..."
git clone https://www.modelscope.cn/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8.git

echo "Cloning Qwen3-Coder-480B-A35B-Instruct-FP8..."
git clone https://www.modelscope.cn/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8.git

MOE
https://pan.baidu.com/s/1gcjUMDz_vcmLS6yZAeH35Q?pwd=pbo8
https://zhuanlan.zhihu.com/p/719466709
https://triton.csdn.net/6719f57fcd8b2677c3d4ab94.html
https://zhuanlan.zhihu.com/p/21251657579
https://zhuanlan.zhihu.com/p/21251657579
https://zhuanlan.zhihu.com/p/25401744621
https://zhuanlan.zhihu.com/p/1895178845830771205
https://blog.csdn.net/zpp13hao1/article/details/147891337
https://zhuanlan.zhihu.com/p/1911059432953061899
https://mmssai.com/archives/33850
https://www.cnblogs.com/cavalier-chen/p/18937098
https://zhuanlan.zhihu.com/p/26436168971
https://zhuanlan.zhihu.com/p/1895178845830771205
https://blog.csdn.net/zpp13hao1/article/details/147891337
https://mmssai.com/archives/33850
echo "所有操作完成！"



nvshmem
https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html
https://zhuanlan.zhihu.com/p/26575363781
https://blog.csdn.net/jkjgj/article/details/142066495
https://zhuanlan.zhihu.com/p/1952325639211315297
https://zhuanlan.zhihu.com/p/1941562551407187752
https://zhuanlan.zhihu.com/p/1933899894881489690
https://blog.csdn.net/jkjgj/article/details/142066495
https://zhuanlan.zhihu.com/p/26082845081
https://www.51cto.com/aigc/6550.html
https://blog.csdn.net/gitblog_00341/article/details/151640319



__device__ __forceinline__ float gelu(const float& val) {
  constexpr float kAlpha = M_SQRT1_2;
  return val * 0.5f * (1.0f + ::erf(val * kAlpha));
}

__device__ __forceinline__ float gelu_tanh(const float& val) {
  const float cdf =
      0.5f * (1.0f + math::tanh((0.7978845608028654f * (val + 0.044715f * val * val * val))));
  return val * cdf;
}

void silu_and_mul(at::Tensor& out, at::Tensor& input, bool enable_pdl) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);

  const c10::cuda::OptionalCUDAGuard device_guard(out.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;
    auto kernel = flashinfer::activation::act_and_mul_kernel<c_type, silu>;

    cudaLaunchKernelEx(&config, kernel, static_cast<c_type*>(out.data_ptr()),
                       static_cast<c_type*>(input.data_ptr()), d);

namespace activation {
template <typename T, float (*Activation)(const float&)>
__global__ void act_and_mul_kernel(T* __restrict__ out, const T* __restrict__ input, const int d) {
  constexpr uint32_t vec_size = 16 / sizeof(T);
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * 2 * d;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    vec_t<float, vec_size> x_vec, y_vec, out_vec;
    x_vec.cast_load(input + offset + idx * vec_size);
    y_vec.cast_load(input + offset + d + idx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      out_vec[i] = Activation(x_vec[i]) * y_vec[i];
    }
    out_vec.cast_store(out + token_idx * d + idx * vec_size);
  }

  const int64_t remaining_offset = d - d % (stride * vec_size);
  // process the remaining elements
#pragma unroll 1
  for (int64_t idx = thread_idx; idx < d % (stride * vec_size); idx += stride) {
    float x = input[offset + remaining_offset + idx],
          y = input[offset + remaining_offset + d + idx];
    out[token_idx * d + remaining_offset + idx] = Activation(x) * y;
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

__device__ __forceinline__ __nv_bfloat162 silu_mul_half2(const __nv_bfloat162& val, const __nv_bfloat162& factor ) {
      return __hmul2(__h2div(val, __hadd2(__float2bfloat162_rn(1.0f), h2exp(__hneg2(val)))), factor);
}
__global__ void act_mul_f32_kernel(float* __restrict__ out, const float* __restrict__ input, const int d) {
        const int64_t vec_size = 4;
    const int64_t token_idx = blockIdx.x;
    const int64_t thread_idx = threadIdx.x;
    const int64_t stride = blockDim.x;
    const int64_t offset = token_idx * 2 * d;

    cudaDeviceSynchronize(); 
    cudaError_t err = cudaGetLastError(); 
    if (err != cudaSuccess) { 
        printf("CUDA Error: %s\n",cudaGetErrorString(err)); 
    }



# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert TRT-LLM GPTQ format checkpoint to Kimi-K2-Thinking compressed-tensors format.
The source GPTQ checkpoint uses:
- int32 packing: 8 int4 values per int32
- Requires .qweight, .scales, .qzeros tensors
TRT-LLM compressed-tensors format uses:
- int32 packing: 8 int4 values per int32 (same)
- group_size: 32
- symmetric quantization
"""

import argparse
import json
from pathlib import Path

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm


def unpack_int32_to_int4_gptq(weight_packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack GPTQ int32 tensor containing 8 int4 values into int8 tensor.
    Args:
        weight_packed: Shape (K/8, N) dtype int32
    Returns:
        unpacked: Shape (K, N) dtype int8 with values in range [-8, 7]
    """
    # Convert int32 to uint8 view to extract nibbles
    w_packed_uint8 = weight_packed.contiguous().view(torch.uint8)

    # Each int32 = 4 bytes, each byte has 2 int4 values
    k_div_8, n = weight_packed.shape
    w_packed_uint8 = w_packed_uint8.view(k_div_8 * 4, n)

    # Allocate output: (K, N) where K = K_div_8 * 8
    k = k_div_8 * 8
    w_unpacked = torch.zeros(k, n, dtype=torch.int8)

    # Extract low and high nibbles
    w_unpacked[0::2, :] = (w_packed_uint8 & 0x0F).to(torch.int8)
    w_unpacked[1::2, :] = (w_packed_uint8 >> 4).to(torch.int8)

    # Convert from uint4 [0, 15] to int4 [-8, 7]
    # Values > 7 should be interpreted as negative
    w_unpacked[w_unpacked > 7] -= 16

    return w_unpacked.contiguous()


def pack_int4_to_int32_compressed(weight_unpacked: torch.Tensor) -> torch.Tensor:
    """
    Pack int8 tensor (with int4 values) into int32 compressed-tensors format.
    Args:
        weight_unpacked: Shape (N, K) dtype int8 with values in range [-8, 7]
    Returns:
        packed: Shape (N, K/8) dtype int32
    """
    n, k = weight_unpacked.shape
    assert k % 8 == 0, "K must be divisible by 8"

    # Convert int4 [-8, 7] to uint4 [0, 15]
    w_uint4 = weight_unpacked.clone()
    w_uint4[w_uint4 < 0] += 16
    w_uint4 = w_uint4.to(torch.uint8)

    # Pack 2 uint4 into 1 uint8
    w_packed_uint8 = torch.zeros(n, k // 2, dtype=torch.uint8)
    w_packed_uint8 = (w_uint4[:, 1::2] << 4) | (w_uint4[:, 0::2])

    # Reshape to int32: (N, K/2) uint8 -> (N, K/8, 4) uint8 -> (N, K/8) int32
    w_packed_int32 = (
        w_packed_uint8.view(n, k // 8, 4).contiguous().view(torch.uint8).view(n, k // 8).view(torch.int32)
    )

    return w_packed_int32.contiguous()


def convert_gptq_to_compressed_tensor(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Convert GPTQ format to compressed-tensors format.
    Args:
        qweight: Shape (K/8, N) dtype int32 (GPTQ format is transposed)
        scales: Shape (K/group_size, N) dtype fp16 (GPTQ format is transposed)
        qzeros: Shape (K/group_size, N/8) dtype int32 (not used for symmetric)
        group_size: Quantization group size
    Returns:
        Dictionary with:
            - weight_packed: Shape (N, K/8) dtype int32
            - weight_scale: Shape (N, K/group_size) dtype bfloat16
            - weight_shape: Shape (2,) dtype int64 containing [N, K]
    """
    k_div_8, n = qweight.shape
    k = k_div_8 * 8

    # GPTQ weights are transposed: (K/8, N) -> Need to transpose back to (N, K/8)
    # First unpack to (K, N), then transpose to (N, K), then repack to (N, K/8)
    
    # Unpack GPTQ format
    weight_unpacked = unpack_int32_to_int4_gptq(qweight)  # Shape: (K, N)
    
    # Transpose to compressed-tensors layout
    weight_unpacked = weight_unpacked.t().contiguous()  # Shape: (N, K)
    
    # Repack in compressed-tensors format
    weight_packed = pack_int4_to_int32_compressed(weight_unpacked)  # Shape: (N, K/8)

    # Transpose scales: (K/group_size, N) -> (N, K/group_size)
    weight_scale = scales.t().contiguous().to(torch.bfloat16)

    # Store original weight shape
    weight_shape = torch.tensor([n, k], dtype=torch.int64)

    return {
        "weight_packed": weight_packed,
        "weight_scale": weight_scale,
        "weight_shape": weight_shape,
    }


def convert_checkpoint(
    input_dir: str,
    output_dir: str,
    num_shards: int | None = None,
    skip_existing: bool = True,
):
    """
    Convert all shards from GPTQ to compressed-tensors format.
    Args:
        input_dir: Source checkpoint directory (GPTQ format)
        output_dir: Output checkpoint directory (compressed-tensors format)
        num_shards: Number of shards to process (None = all)
        skip_existing: Skip conversion if output shard already exists
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all safetensors files
    shard_files = sorted(input_path.glob("model-*.safetensors"))
    if not shard_files:
        raise ValueError(f"No model shards found in {input_dir}")

    if num_shards is not None:
        shard_files = shard_files[:num_shards]

    print(f"Found {len(shard_files)} shards to process")

    # Track weight mapping for index file
    new_weight_map = {}

    # Load and convert each shard
    for shard_idx, shard_file in enumerate(tqdm(shard_files, desc="Processing shards")):
        shard_name = shard_file.name
        output_file = output_path / shard_name

        # Check if output already exists
        if skip_existing and output_file.exists():
            print(f"\n⏭️  Skipping {shard_name} (already exists)")
            # Still need to build the weight_map from existing file
            with safetensors.safe_open(str(output_file), framework="pt", device="cpu") as f:
                for key in f:
                    new_weight_map[key] = shard_name
            continue

        print(f"\n🔄 Converting {shard_file.name}...")

        # Load source shard
        source_tensors = {}
        with safetensors.safe_open(str(shard_file), framework="pt", device="cpu") as f:
            for key in f:
                source_tensors[key] = f.get_tensor(key)

        # Convert tensors
        output_tensors = {}

        # Group GPTQ tensors by base key
        processed_keys = set()
        
        for key in tqdm(source_tensors.keys(), desc="Converting tensors", leave=False):
            if key in processed_keys:
                continue
                
            if key.endswith(".qweight"):
                # This is a quantized weight - convert to compressed-tensors format
                base_key = key[: -len(".qweight")]
                scales_key = base_key + ".scales"
                qzeros_key = base_key + ".qzeros"

                if scales_key in source_tensors and qzeros_key in source_tensors:
                    # Convert to compressed-tensors format
                    compressed_tensors = convert_gptq_to_compressed_tensor(
                        qweight=source_tensors[key],
                        scales=source_tensors[scales_key],
                        qzeros=source_tensors[qzeros_key],
                        group_size=32,
                    )

                    # Save with compressed-tensors naming convention
                    packed_key = base_key + ".weight_packed"
                    scale_key = base_key + ".weight_scale"
                    shape_key = base_key + ".weight_shape"

                    output_tensors[packed_key] = compressed_tensors["weight_packed"]
                    output_tensors[scale_key] = compressed_tensors["weight_scale"]
                    output_tensors[shape_key] = compressed_tensors["weight_shape"]

                    new_weight_map[packed_key] = shard_name
                    new_weight_map[scale_key] = shard_name
                    new_weight_map[shape_key] = shard_name

                    # Mark related keys as processed
                    processed_keys.add(key)
                    processed_keys.add(scales_key)
                    processed_keys.add(qzeros_key)
                else:
                    print(f"Warning: Missing scales or qzeros for {key}")

            elif key.endswith((".scales", ".qzeros")):
                # Skip these as they're handled above
                processed_keys.add(key)
                continue
            else:
                # Keep non-quantized tensors as-is
                output_tensors[key] = source_tensors[key]
                new_weight_map[key] = shard_name
                processed_keys.add(key)

        # Save converted shard
        safetensors.torch.save_file(output_tensors, str(output_file))
        print(f"✅ Saved to {output_file}")

    # Copy config.json and update quantization settings
    config_file = input_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

        # Remove TRT-LLM quantization config if present
        config.pop("quantization", None)

        # Add HuggingFace compressed-tensors quantization config
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "group_size": 32,
            "num_bits": 4,
            "strategy": "channel",
            "type": "symmetric",
        }

        output_config_file = output_path / "config.json"
        with open(output_config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nSaved config to {output_config_file}")

    # Generate new safetensors index file
    index_data = {
        "metadata": {
            "total_size": sum(
                (output_path / shard_file.name).stat().st_size for shard_file in shard_files
            )
        },
        "weight_map": new_weight_map,
    }

    index_file = output_path / "model.safetensors.index.json"
    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"\nGenerated index file: {index_file}")
    print(f"  Total tensors: {len(new_weight_map)}")

    # Copy other necessary files
    import shutil

    # JSON files (tokenizer and generation config)
    for file in [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    # Python files (model architecture, custom tokenizers)
    for file in ["configuration_deepseek.py", "modeling_deepseek.py", "tokenization_kimi.py"]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    # Tokenizer model files
    for file in ["tiktoken.model", "tokenizer.model", "sentencepiece.model"]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    # Template files
    for file in ["chat_template.jinja", "chat_template.json"]:
        src = input_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            print(f"Copied: {file}")

    print(f"\n✓ Conversion complete! Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GPTQ checkpoint to Kimi-K2 compressed-tensors format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/scratch.omniml_data_2/zhiyuc/checkpoints/Kimi-K2-Thinking-GPTQ",
        help="Input checkpoint directory with GPTQ format",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/scratch.omniml_data_1/models/Kimi-K2-Thinking-Compressed",
        help="Output directory for compressed-tensors format checkpoint",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of shards to convert (default: all)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-convert shards even if they already exist (default: skip existing)",
    )

    args = parser.parse_args()

    convert_checkpoint(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
