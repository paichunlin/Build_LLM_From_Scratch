import torch
import triton
import triton.language as tl

def get_flashattention_autograd_function_triton() -> Type:
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,):
        # Program indices
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        max_neg_value = -3.4028234663852886e+38

        # Offset each pointer with the corresponding batch index
        # multiplied with the batch stride for each tensor
        Q_block_ptr = tl.make_block_ptr(
                    Q_ptr + batch_index * stride_qb,
                    shape=(N_QUERIES, D),
                    strides=(stride_qq, stride_qd),
                    offsets=(query_tile_index * Q_TILE_SIZE, 0),
                    block_shape=(Q_TILE_SIZE, D),
                    order=(1, 0),)

        O_block_ptr = tl.make_block_ptr(
                    O_ptr + batch_index * stride_ob,
                    shape=(N_QUERIES, D),
                    strides=(stride_oq, stride_od),
                    offsets=(query_tile_index * Q_TILE_SIZE, 0),
                    block_shape=(Q_TILE_SIZE, D),
                    order=(1, 0),)

        L_block_ptr = tl.make_block_ptr(
                    L_ptr + batch_index * stride_lb,
                    shape=(N_QUERIES,),
                    strides=(stride_lq,),
                    offsets=(query_tile_index * Q_TILE_SIZE,),
                    block_shape=(Q_TILE_SIZE,),
                    order=(0,),)

        K_block_ptr = tl.make_block_ptr(
                    K_ptr + batch_index * stride_kb,
                    shape=(N_KEYS, D),
                    strides=(stride_kk, stride_kd),
                    offsets=(query_tile_index * K_TILE_SIZE, 0),
                    block_shape=(K_TILE_SIZE, D),
                    order=(1, 0),)

        V_block_ptr = tl.make_block_ptr(
                    V_ptr + batch_index * stride_vb,
                    shape=(N_KEYS, D),
                    strides=(stride_vk, stride_vd),
                    offsets=(query_tile_index * K_TILE_SIZE, 0),
                    block_shape=(K_TILE_SIZE, D),
                    order=(1, 0),)

        output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        qc = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
        row_sums = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
        row_maxes = tl.full((Q_TILE_SIZE, 1), max_neg_value, dtype=tl.float32)

        for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
          kc = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
          vc = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
          attn_weights = tl.dot(qc, tl.trans(kc)) * scale
          block_row_maxes = tl.max(attn_weights, axis = -1, keep_dims = True)
          new_row_maxes = tl.maximum(block_row_maxes, row_maxes)

          exp_weights = tl.exp(attn_weights - new_row_maxes)
          block_row_sums = tl.sum(exp_weights, axis = -1, keep_dims = True)

          exp_values = tl.dot(exp_weights, vc)
          exp_row_max_diff = tl.exp(row_maxes - new_row_maxes)

          row_sums = exp_row_max_diff * row_sums + block_row_sums
          output = output * exp_row_max_diff + exp_values
          row_maxes = new_row_maxes * 1

          K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
          V_block_ptr = V_block_ptr.advance((0, K_TILE_SIZE))

        tl.store(O_block_ptr, output/row_sums, boundary_check=(0, 1))
        tl.store(L_block_ptr, tl.log(row_sums) + row_maxes, boundary_check=(0, 1))

    class TritonFlashAttention2(torch.autograd.Function):
      @staticmethod
      def forward(ctx, q, k, v, is_causal=False):
        n_queries, D, n_keys = q.shape[-2], q.shape[-1], k.shape[-2]
        scale = D ** -0.5
        output = torch.empty_like(q, device=q.device)
        logsumexp = torch.empty((*q.shape[:-1], 1), device=q.device)

        ctx.D = D
        ctx.K_TILE_SIZE = 16 # Each thread processes 16 batch elements at a time
        ctx.Q_TILE_SIZE = 16 # Each thread processes 16 batch elements at a time

        grid = (triton.cdiv(n_queries, ctx.Q_TILE_SIZE),)
        flash_fwd_kernel[grid](
          q, k, v,
          output, logsumexp,
          q.stride(0), q.stride(1), q.stride(2),
          k.stride(0), k.stride(1), k.stride(2),
          v.stride(0), v.stride(1), v.stride(2),
          output.stride(0), output.stride(1), output.stride(2),
          logsumexp.stride(0), logsumexp.stride(1),
          n_queries, n_keys,
          scale,
          D=ctx.D,
          Q_TILE_SIZE=ctx.Q_TILE_SIZE,
          K_TILE_SIZE=ctx.K_TILE_SIZE,
        )

        ctx.save_for_backward(q, k, v, output, logsumexp.squeeze())
        return output

      @staticmethod
      def backward(ctx, grad_output):
        raise NotImplementedError

    return TritonFlashAttention2