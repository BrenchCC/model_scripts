# 使用方法:
#   bash vllm_infer.sh [选项]
#
# 选项说明 (不填则使用默认值):
#   -m, --model_path <path>       模型路径
#                                 默认: ../models/Qwen3-VL-8B-Instruct
#   -p, --port <port>             服务端口
#                                 默认: 8888
#   --host <host>                 服务监听地址
#                                 默认: ::
#   --dtype <dtype>               模型数据类型 (auto/float16/bfloat16)
#                                 默认: auto
#   -g, --gpu-memory-utilization <f>  GPU 显存占用比例 (0.0 - 1.0)
#                                     默认: 0.7
#   -t, --tensor-parallel-size <n>    Tensor 并行大小 (通常等于使用的 GPU 卡数)
#                                     默认: 2
#   -c, --cuda-visible-devices <ids>  指定使用的 GPU 设备 ID (例如 "0,1")
#                                     默认: 不指定 (使用所有可用 GPU)
#   -h, --help                    显示本帮助信息并退出
#
# 示例:
#   1. 使用默认参数启动:
#      bash vllm_infer.sh
#
#   2. 完整自定义参数 (展示所有选项):
# bash infer/vllm_infer.sh \
#     --model_path ../models/Qwen3-VL-8B-Instruct \
#     --port 7777 \
#     --host :: \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.9 \
#     --tensor-parallel-size 1 \
#     --cuda-visible-devices "2"
#
# 切换到脚本所在目录 (可选)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
# pip install numpy==1.26.4
# ------------------------------------------------------------------------------
# 1. 默认参数配置
# ------------------------------------------------------------------------------
# MODEL_PATH="../models/hf_ckpt"
MODEL_PATH="../models/Qwen3-VL-8B-Instruct"
PORT=8888
HOST="::"
DTYPE="auto"
GPU_MEMORY_UTILIZATION=0.7
TENSOR_PARALLEL_SIZE=2
CUDA_VISIBLE_DEVICES_ARG="0,1"
# vLLM 本地媒体目录白名单；设为 "/" 表示允许访问整机路径（按进程权限）。
ALLOWED_LOCAL_MEDIA_PATH="/"

# ------------------------------------------------------------------------------
# 2. 帮助函数
# ------------------------------------------------------------------------------
usage() {
    # 打印脚本头部的注释作为帮助信息
    sed -n '2,36p' "$0"
    exit 1
}

# ------------------------------------------------------------------------------
# 3. 解析命令行参数
# ------------------------------------------------------------------------------
# 使用 getopt 解析长短参数
ARGS=$(getopt -o m:p:g:t:c:h --long model_path:,port:,host:,dtype:,gpu-memory-utilization:,tensor-parallel-size:,cuda-visible-devices:,help -- "$@")

# 检查参数解析是否成功
if [ $? -ne 0 ]; then
    usage
fi

# 重置位置参数
eval set -- "$ARGS"

while true; do
    case "$1" in
        -m|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        -g|--gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        -t|--tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        -c|--cuda-visible-devices)
            CUDA_VISIBLE_DEVICES_ARG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "错误: 未知选项 $1"
            usage
            ;;
    esac
done

# ------------------------------------------------------------------------------
# 4. 启动服务
# ------------------------------------------------------------------------------
echo "========================================"
echo "正在启动 vllm serve..."
echo "----------------------------------------"
echo "配置详情:"
echo "  模型路径:       $MODEL_PATH"
echo "  端口:           $PORT"
echo "  主机:           $HOST"
echo "  数据类型:       $DTYPE"
echo "  显存利用率:     $GPU_MEMORY_UTILIZATION"
echo "  Tensor并行数:   $TENSOR_PARALLEL_SIZE"
if [ -n "$CUDA_VISIBLE_DEVICES_ARG" ]; then
    echo "  指定显卡:       $CUDA_VISIBLE_DEVICES_ARG"
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ARG"
fi
echo "----------------------------------------"
echo "日志文件:       vllm.log"
echo "========================================"

# 使用 nohup 后台启动，并重定向标准输出和错误输出到日志文件
nohup vllm serve \
 "$MODEL_PATH" \
 --port "$PORT" \
 --host "$HOST" \
 --dtype "$DTYPE" \
 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
 --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
 --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" > vllm.log 2>&1 &

PID=$!
echo "vllm 服务已在后台启动 (PID: $PID)"
echo "你可以使用以下命令查看实时日志:"
echo "tail -f vllm.log"