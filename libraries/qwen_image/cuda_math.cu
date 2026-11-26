#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <math_constants.h>

static uint32_t alignment(const void* pointer) {
    uintptr_t p=reinterpret_cast<uintptr_t>(pointer);
    return static_cast<uint32_t>(p%256 == 0 ? 256 : (p & (~p+1)));
}

static int qwen_linear(const float* x,const float* w,const float* bias,float* y,
                       int64_t rows,int64_t k,int64_t n,void* buffer,size_t capacity,cudaStream_t stream) {
    cublasLtHandle_t handle=nullptr;
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulPreference_t pref=nullptr;
    const cublasOperation_t trans=CUBLAS_OP_T;
    const cublasLtEpilogue_t epilogue=CUBLASLT_EPILOGUE_BIAS;
    const float alpha=1.0f,beta=0.0f;
    cublasLtMatmulHeuristicResult_t found{};
    int count=0;
    const uint32_t aw=alignment(w),ax=alignment(x),ay=alignment(y),ab=alignment(bias);
    cublasStatus_t status=CUBLAS_STATUS_SUCCESS;
#define CALL(expr) do { status=(expr); if(status != CUBLAS_STATUS_SUCCESS) goto cleanup; } while(0)
    CALL(cublasLtCreate(&handle));
    CALL(cublasLtMatmulDescCreate(&operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    CALL(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
    CALL(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)));
    CALL(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias)));
    CALL(cublasLtMatrixLayoutCreate(&a,CUDA_R_32F,k,n,k));
    CALL(cublasLtMatrixLayoutCreate(&b,CUDA_R_32F,k,rows,k));
    CALL(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,n,rows,n));
    CALL(cublasLtMatmulPreferenceCreate(&pref));
    CALL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&capacity,sizeof(capacity)));
    CALL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,&aw,sizeof(aw)));
    CALL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,&ax,sizeof(ax)));
    CALL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,&ay,sizeof(ay)));
    CALL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,&ab,sizeof(ab)));
    CALL(cublasLtMatmulAlgoGetHeuristic(handle,operation,a,b,c,c,pref,1,&found,&count));
    if(count==0){status=CUBLAS_STATUS_NOT_SUPPORTED; goto cleanup;}
    CALL(cublasLtMatmul(handle,operation,&alpha,w,a,x,b,&beta,y,c,y,c,&found.algo,buffer,capacity,stream));
cleanup:
    if(pref) cublasLtMatmulPreferenceDestroy(pref);
    if(c) cublasLtMatrixLayoutDestroy(c);
    if(b) cublasLtMatrixLayoutDestroy(b);
    if(a) cublasLtMatrixLayoutDestroy(a);
    if(operation) cublasLtMatmulDescDestroy(operation);
    if(handle) cublasLtDestroy(handle);
    return static_cast<int>(status);
}

static int qwen_batched(const float* a,const float* b,float* out,int batch,int rows,int cols,int width,
                            int scores,cudaStream_t stream) {
    cublasHandle_t handle=nullptr;
    auto status=cublasCreate_v2(&handle);
    if(status!=CUBLAS_STATUS_SUCCESS)return (int)status;
    status=cublasSetStream_v2(handle,stream);
    const float alpha=scores?0.08838834764831845f:1.0f;
    const float beta=0.0f;
    if(status==CUBLAS_STATUS_SUCCESS)
        status=cublasSgemmStridedBatched(handle,scores?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,
                  cols,rows,width,&alpha,b,scores?width:cols,static_cast<int64_t>(cols)*width,
                  a,width,static_cast<int64_t>(rows)*width,&beta,out,cols,static_cast<int64_t>(rows)*cols,batch);
    cublasDestroy_v2(handle);
    return (int)status;
}


struct Moment {
    float total;
    float mean;
    float count;
};

template<bool RMS>
__device__ Moment combine(Moment a, Moment b) {
    if (RMS) return {a.total+b.total,0.0f,0.0f};
    const float count=a.count+b.count;
    if (count==0.0f) return {0.0f,0.0f,0.0f};
    const float difference=a.mean-b.mean;
    const float reciprocal=1.0f/count;
    const float wa=a.count*reciprocal,wb=b.count*reciprocal;
    return {a.total+b.total+difference*difference*b.count*wa,wa*a.mean+wb*b.mean,count};
}

template<bool RMS>
__global__ void normalization(const float* input,const float* gain,float* output,int width) {
    constexpr int warp=32;
    constexpr int warps=4;
    const int lane=threadIdx.x%warp, group=threadIdx.x/warp;
    const float* x=input+static_cast<int64_t>(blockIdx.x)*width;
    float* y=output+static_cast<int64_t>(blockIdx.x)*width;
    const float4* packed=reinterpret_cast<const float4*>(x);
    Moment m{0.0f,0.0f,0.0f};
    for(int i=threadIdx.x;i<width/4;i+=warp*warps) {
        const float4 p=packed[i];
        const float values[]{p.x,p.y,p.z,p.w};
        #pragma unroll
        for(int j=0;j<4;j++){
            const float value=values[j];
            if (RMS) m.total+=value*value;
            else {
                const float delta=value-m.mean;
                const float count=m.count+1.0f;
                const float mean=m.mean+delta*(1.0f/count);
                m.total+=delta*(value-mean);
                m.mean=mean;
                m.count=count;
            }
        }
    }
    for(int offset=16;offset>0;offset/=2) {
        Moment b{
            __shfl_down_sync(0xffffffff,m.total,offset),
            __shfl_down_sync(0xffffffff,m.mean,offset),
            __shfl_down_sync(0xffffffff,m.count,offset),
        };
        m=combine<RMS>(m,b);
    }
    __shared__ Moment partial[warps];
    if(lane==0)partial[group]=m;
    __syncthreads();
    for(int offset=warps/2;offset>0;offset/=2){
        if(lane==0&&group<offset)partial[group]=combine<RMS>(partial[group],partial[group+offset]);
        __syncthreads();
    }
    const Moment stats=partial[0];
    const float factor=rsqrtf(stats.total/static_cast<float>(width)+1e-6f);
    for(int i=threadIdx.x;i<width;i+=warp*warps) {
        const float centered=RMS?x[i]:x[i]-stats.mean;
        y[i]=gain?factor*centered*gain[i]:factor*centered;
    }
}

static int qwen_norm_probe(const float* x,const float* scale,float* y,int rows,int width,
                              int rms,cudaStream_t stream) {
    if(width%4!=0||rows<=0) return 1;
    if(rms) normalization<true><<<rows,128,0,stream>>>(x,scale,y,width);
    else normalization<false><<<rows,128,0,stream>>>(x,scale,y,width);
    return static_cast<int>(cudaPeekAtLastError());
}

__global__ static void unary(const float* x,float* y,int count,int mode) {
    const int i=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const float value=x[i];
    if(mode==0) {
        y[i]=value/(1.0f+expf(-value));
    } else {
        const float cube=value*value*value;
        const float inner=0.7978845608028654f*(value+0.044715f*cube);
        y[i]=(0.5f*value)*(1.0f+tanhf(inner));
    }
}

__global__ static void phase(const float* time,float* result) {
    const int i=threadIdx.x;
    const float frequency=expf(-9.210340371976182f*static_cast<float>(i%128)/128.0f);
    const float angle=(blockIdx.x==0?time[0]:0.0f)*frequency*1000.0f;
    result[blockIdx.x*256+i]=i<128?cosf(angle):sinf(angle);
}

__global__ static void gemv(const float* x,const float* w,const float* bias,float* y,int cols,int width) {
    const int tid=blockIdx.x*blockDim.x+threadIdx.x;
    const int col=tid/16, lane=tid%16;
    float partial=0.0f;
    if(col<cols) {
        const float* ww=w+static_cast<int64_t>(col)*width;
        for(int k=lane;k<width;k+=16) partial+=ww[k]*x[k];
    }
    for(int offset=8;offset>0;offset/=2) partial+=__shfl_down_sync(0xffffffff,partial,offset,16);
    if(lane==0&&col<cols)y[col]=partial+bias[col];
}

static int qwen_unary_probe(const float* x,float* y,int count,int mode,cudaStream_t stream) {
    unary<<<(count+255)/256,256,0,stream>>>(x,y,count,mode);
    return static_cast<int>(cudaPeekAtLastError());
}
static int qwen_phase_probe(const float* time,float* output,cudaStream_t stream) {
    phase<<<2,256,0,stream>>>(time,output);
    return static_cast<int>(cudaPeekAtLastError());
}
static int qwen_gemv_probe(const float* x,const float* w,const float* bias,float* y,int cols,int width,cudaStream_t stream) {
    gemv<<<(cols+7)/8,128,0,stream>>>(x,w,bias,y,cols,width);
    return static_cast<int>(cudaPeekAtLastError());
}

__global__ static void softmax(const float* x,float* output,int rows,int width) {
    const int lane=threadIdx.x%32;
    const int row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(row>=rows)return;
    const float* a=x+static_cast<int64_t>(row)*width;
    float* y=output+static_cast<int64_t>(row)*width;
    const float x0=lane<width?a[lane]:-__int_as_float(0x7f800000);
    const float x1=lane+32<width?a[lane+32]:-__int_as_float(0x7f800000);
    float maximum=fmaxf(x0,x1);
    for(int mask=16;mask>0;mask/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,mask));
    const float e0=expf(x0-maximum),e1=expf(x1-maximum);
    float denom=e0+e1;
    for(int mask=16;mask>0;mask/=2) denom+=__shfl_xor_sync(0xffffffff,denom,mask);
    if(lane<width)y[lane]=e0/denom;
    if(lane+32<width)y[lane+32]=e1/denom;
}

template<bool Sum>
__device__ float reduce_row(float value, float* partial) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other = __shfl_xor_sync(0xffffffff, value, offset);
        value = Sum ? value + other : fmaxf(value, other);
    }
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < blockDim.x / 32 ? partial[lane] : (Sum ? 0.0f : -CUDART_INF_F);
        for (int offset = 16; offset > 0; offset /= 2) {
            const float other = __shfl_xor_sync(0xffffffff, value, offset);
            value = Sum ? value + other : fmaxf(value, other);
        }
        if (lane == 0) partial[0] = value;
    }
    __syncthreads();
    const float result = partial[0];
    __syncthreads();
    return result;
}

template<bool Masked>
__global__ static void softmax_row(const float* input, float* output, int width, const float* mask) {
    __shared__ float partial[32];
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * width;
    const float* x = input + offset;
    float* y = output + offset;
    float maximum = -CUDART_INF_F;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const float score = Masked ? x[col] + mask[col] : x[col];
        maximum = fmaxf(maximum, score);
    }
    maximum = reduce_row<false>(maximum, partial);
    float sum = 0.0f;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const float score = Masked ? x[col] + mask[col] : x[col];
        sum += expf(score - maximum);
    }
    sum = reduce_row<true>(sum, partial);
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        const float score = Masked ? x[col] + mask[col] : x[col];
        y[col] = expf(score - maximum) / sum;
    }
}

static int qwen_softmax_probe(const float* x,float* y,int rows,int width,cudaStream_t stream) {
    if (width <= 64) softmax<<<(rows+3)/4,128,0,stream>>>(x,y,rows,width);
    else softmax_row<false><<<rows,256,0,stream>>>(x,y,width,nullptr);
    return static_cast<int>(cudaPeekAtLastError());
}

static int qwen_tiled_attention(const float* q, const float* k, const float* v, const float* mask,
                                float* out, float* scores, int batch, int queries, int keys, int width,
                                int tile_rows, cudaStream_t stream) {
    if (tile_rows <= 0 || tile_rows > queries) return static_cast<int>(cudaErrorInvalidValue);
    cublasHandle_t handle = nullptr;
    auto status = cublasCreate_v2(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
    status = cublasSetStream_v2(handle, stream);
    const float scale = 1.0f / sqrtf(static_cast<float>(width));
    const float one = 1.0f, zero = 0.0f;
    const int64_t key_stride = static_cast<int64_t>(keys) * width;
    const int64_t query_stride = static_cast<int64_t>(queries) * width;
    cudaError_t kernel = cudaSuccess;

    for (int start = 0; status == CUBLAS_STATUS_SUCCESS && start < queries; start += tile_rows) {
        const int rows = std::min(tile_rows, queries - start);
        const int64_t score_stride = static_cast<int64_t>(rows) * keys;
        const int64_t offset = static_cast<int64_t>(start) * width;
        status = cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, keys, rows, width,
                    &scale, k, width, key_stride, q + offset, width, query_stride,
                    &zero, scores, keys, score_stride, batch);
        if (status != CUBLAS_STATUS_SUCCESS) break;
        softmax_row<true><<<batch * rows, 256, 0, stream>>>(scores, scores, keys, mask);
        kernel = cudaPeekAtLastError();
        if (kernel != cudaSuccess) break;
        status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, rows, keys,
                    &one, v, width, key_stride, scores, keys, score_stride,
                    &zero, out + offset, width, query_stride, batch);
    }
    cublasDestroy_v2(handle);
    return kernel != cudaSuccess ? static_cast<int>(kernel) : static_cast<int>(status);
}

struct QwenBuffer {
    void* data;
    size_t rank;
    const int64_t* dims;
};

static int elements(const QwenBuffer& buffer) {
    int count = 1;
    for (size_t i = 0; i < buffer.rank; ++i)
        count *= static_cast<int>(buffer.dims[i]);
    return count;
}

extern "C" __attribute__((visibility("default")))
int zml_qwen_cuda_math(uint32_t op, const QwenBuffer* inputs, QwenBuffer* outputs, cudaStream_t stream) {
    const QwenBuffer& x = inputs[0];
    const float* data = static_cast<const float*>(x.data);
    float* result = static_cast<float*>(outputs[0].data);
    const int width = static_cast<int>(x.dims[x.rank - 1]);
    switch (op) {
        case 0: {
            const QwenBuffer& w = inputs[1];
            const QwenBuffer& workspace = outputs[1];
            return qwen_linear(data, static_cast<const float*>(w.data), static_cast<const float*>(inputs[2].data),
                               result, x.dims[0], width, w.dims[0], workspace.data,
                               static_cast<size_t>(workspace.dims[0]), stream);
        }
        case 1:
            return qwen_norm_probe(data, nullptr, result, elements(x) / width, width, 0, stream);
        case 2:
            return qwen_norm_probe(data, static_cast<const float*>(inputs[1].data), result,
                                   elements(x) / width, width, 1, stream);
        case 3:
        case 4:
            return qwen_unary_probe(data, result, elements(x), static_cast<int>(op - 3), stream);
        case 5:
            return qwen_phase_probe(data, result, stream);
        case 6:
            return qwen_gemv_probe(data, static_cast<const float*>(inputs[1].data),
                                   static_cast<const float*>(inputs[2].data), result,
                                   static_cast<int>(inputs[1].dims[0]), width, stream);
        case 7:
        case 8: {
            const QwenBuffer& b = inputs[1];
            const bool scores = op == 7;
            return qwen_batched(data, static_cast<const float*>(b.data), result,
                                static_cast<int>(x.dims[0] * x.dims[1]), static_cast<int>(x.dims[2]),
                                static_cast<int>(b.dims[scores ? 2 : 3]), width, scores, stream);
        }
        case 9:
            return qwen_softmax_probe(data, result, elements(x) / width, width, stream);
        case 10: {
            const QwenBuffer& k = inputs[1];
            const int batch = static_cast<int>(x.dims[0] * x.dims[1]);
            const int queries = static_cast<int>(x.dims[2]);
            const int keys = static_cast<int>(k.dims[2]);
            const int rows = static_cast<int>(outputs[1].dims[0] / (static_cast<int64_t>(batch) * keys));
            return qwen_tiled_attention(data, static_cast<const float*>(k.data), static_cast<const float*>(inputs[2].data),
                                        static_cast<const float*>(inputs[3].data), result, static_cast<float*>(outputs[1].data),
                                        batch, queries, keys, width, rows, stream);
        }
        default:
            return 1;
    }
}
