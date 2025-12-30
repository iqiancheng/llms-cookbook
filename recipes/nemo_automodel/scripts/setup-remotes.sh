#!/usr/bin/env bash
# filename: recipes/scripts/setup-remotes.sh
# 功能：
# - 将现有 "origin" 重命名为 "upstream"（如果存在且 upstream 不存在）
# - 新增或更新 "origin" 指向你的 GitLab 仓库
# - 可重复运行；顶部可改 URL

set -euo pipefail

# 在这里配置你的目标 origin URL
TARGET_ORIGIN_URL="git@gitlab.example.com:username/automodel.git"

# 简单彩色输出
info()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
error() { printf "\033[1;31m[ERROR]\033[0m %s\n" "$*"; }

# 检查是否在 Git 仓库里
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  error "当前目录不是 Git 仓库，请进入仓库根目录后再运行。"
  exit 1
fi

info "读取当前 remotes..."
git remote -v || true

# 如果存在 origin，且 upstream 不存在，则重命名
if git remote | grep -qx "origin"; then
  if git remote | grep -qx "upstream"; then
    warn "'upstream' 已存在，跳过重命名。"
  else
    info "检测到 'origin'，重命名为 'upstream'..."
    git remote rename origin upstream
    info "已将 'origin' 重命名为 'upstream'。"
  fi
else
  warn "未检测到 'origin'，无需重命名。"
fi

# 新增或更新 origin 到目标 URL
if git remote | grep -qx "origin"; then
  CURRENT_URL="$(git remote get-url origin)"
  if [ "$CURRENT_URL" != "$TARGET_ORIGIN_URL" ]; then
    info "更新 'origin' URL 为：$TARGET_ORIGIN_URL"
    git remote set-url origin "$TARGET_ORIGIN_URL"
  else
    info "'origin' 已指向目标地址，无需更新。"
  fi
else
  info "新增 'origin' 指向：$TARGET_ORIGIN_URL"
  git remote add origin "$TARGET_ORIGIN_URL"
fi

info "最终 remotes："
git remote -v

info "完成。现在 'upstream' 是原来的上游（原 origin），你自己的推送地址是 'origin'：$TARGET_ORIGIN_URL"
