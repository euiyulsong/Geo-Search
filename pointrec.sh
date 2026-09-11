cd ~

# 혹시 찌꺼기가 있으면 제거
rm -rf pointrec

# 공식 POINTREC 다운로드
git clone https://github.com/iai-group/sigir2021-pointrec.git pointrec

# 확인
ls -lh pointrec
ls -lh pointrec/relevance
find pointrec/poi_dataset -type f | head
