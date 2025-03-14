#!/bin/bash

# 원격 저장소 URL 설정
REMOTE_URL="https://github.com/Quantum-Intelligence-Frontier/dldna.git"
REMOTE_NAME="qif"

# 현재 브랜치 확인
CURRENT_BRANCH=$(git branch --show-current)

# 올바른 브랜치에 있는지 확인
if [ "$CURRENT_BRANCH" != "draft" ]; then
  echo "Error: You are not on the 'draft' branch."
  exit 1
fi

# 원격 저장소 추가 (이미 존재하는 경우 무시)
if ! git remote | grep -q "$REMOTE_NAME"; then
    git remote add "$REMOTE_NAME" "$REMOTE_URL"
    echo "Remote '$REMOTE_NAME' added."
else:
    echo "Remote '$REMOTE_NAME' already exists."
fi

# 원격 저장소 URL 업데이트
git remote set-url "$REMOTE_NAME" "$REMOTE_URL"

# 원격 저장소의 변경 사항을 가져옴
git fetch "$REMOTE_NAME"


# .gitignore 적용을 위해 캐시 삭제 후 다시 추가
git rm -r --cached .  2>/dev/null # 에러 메시지 숨김 (파일이 없을 경우 대비)
git add .
git commit -m "Apply .gitignore" --allow-empty # 빈 커밋 허용


# 현재 브랜치의 내용을 원격 저장소의 main 브랜치로 강제 push (--force-with-lease 사용)
echo "Pushing current branch '$CURRENT_BRANCH' to $REMOTE_NAME/main (using force-with-lease)..."
git push --force-with-lease "$REMOTE_NAME" "$CURRENT_BRANCH:main"

# 성공 메시지
if [ $? -eq 0 ]; then
  echo "Publish complete (forced update)."
else
  echo "Publish failed. Check for errors above."
fi