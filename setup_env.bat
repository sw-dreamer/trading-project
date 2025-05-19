@echo off
echo SAC 트레이딩 시스템 개발 환경 설정을 시작합니다...

REM Conda 환경 생성 (Python 3.10)
echo Conda 가상환경 생성 중...
call conda create -n sac_trading_py310 python=3.10 -y
call conda activate sac_trading_py310

REM 필수 패키지 설치
echo 필수 패키지 설치 중...
call conda install numpy=1.23.5 -y
#call conda install pytorch=2.1.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
call pip install pandas matplotlib seaborn scikit-learn tqdm requests pytest pytest-cov python-dotenv

REM CUDA 작동 확인
echo CUDA 작동 확인 중...
python -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available()); print('CUDA 버전:', torch.version.cuda); print('GPU 이름:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU 사용 불가')"

echo 환경 설정이 완료되었습니다.
echo 가상환경을 활성화하려면 'conda activate sac_trading_py310' 명령어를 사용하세요. 