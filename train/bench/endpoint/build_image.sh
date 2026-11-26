#!/usr/bin/env bash
# Build and push one model's endpoint image (bench/endpoint/Dockerfile) to ECR; prints the image URI.
#
#   bench/endpoint/build_image.sh kev-4b
#
# Run from anywhere on an x86_64 Linux machine with Docker and AWS credentials (e.g. an EC2 instance in
# ap-northeast-2): the images carry CUDA wheels and vLLM, which are slow to build under emulation on a Mac.
# The tag is {model}-{commit} of train/; build from a clean, committed tree so the tag names the code.
#
# Environment: BASE_IMAGE (default: the training image in train/README.md), REGION, REPOSITORY
# (default: configs/bench.yaml endpoint.ecr_repository).
set -euo pipefail

MODEL="${1:?usage: build_image.sh <model>}"
TRAIN="$(cd "$(dirname "$0")/../.." && pwd)"
REGION="${REGION:-ap-northeast-2}"
REPOSITORY="${REPOSITORY:-ds-jf-decision-model-bench}"
BASE_IMAGE="${BASE_IMAGE:-763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch:2.14-cu133-amzn2023-sagemaker}"

[ -f "$TRAIN/configs/models/$MODEL.yaml" ] || { echo "no model config $MODEL" >&2; exit 1; }
COMMIT="$(git -C "$TRAIN" rev-parse --short=7 HEAD)"
if [ -n "$(git -C "$TRAIN" status --porcelain -- .)" ]; then
  echo "warning: uncommitted changes in train/; the tag $MODEL-$COMMIT will not match the code" >&2
fi
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$REGISTRY/$REPOSITORY:$MODEL-$COMMIT"

# The base image lives in the AWS Deep Learning Containers registry; the result goes to our own.
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${BASE_IMAGE%%/*}" >&2
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY" >&2
aws ecr describe-repositories --region "$REGION" --repository-names "$REPOSITORY" >/dev/null 2>&1 || {
  echo "ECR repository $REPOSITORY does not exist in $REGION; create it first:" >&2
  echo "  aws ecr create-repository --region $REGION --repository-name $REPOSITORY --tags Key=project,Value=ds-jf-decision-model Key=team,Value=ds" >&2
  exit 1
}

docker build --platform linux/amd64 -f "$TRAIN/bench/endpoint/Dockerfile" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg MODEL="$MODEL" -t "$IMAGE" "$TRAIN" >&2
docker push "$IMAGE" >&2
echo "$IMAGE"
