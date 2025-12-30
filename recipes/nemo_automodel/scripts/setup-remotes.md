# 查看现有 remotes
git remote -v

# 将 origin 改名为 upstream（若已存在 upstream 会报错，可先判断）
git remote | grep -qx "upstream" || git remote rename origin upstream

# 新增你的 origin（如果已存在就改 URL）
if git remote | grep -qx "origin"; then
  git remote set-url origin "git@gitlab.example.com:username/automodel.git"
else
  git remote add origin "git@gitlab.example.com:username/automodel.git"
fi

# 验证结果
git remote -v
