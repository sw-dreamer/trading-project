// 티커 심볼에 따라 뉴스를 필터링하는 함수
function filterNews(ticker) {
  // 현재 활성화된 탭 확인
  const activeTabId = document.querySelector('.nav-tabs .nav-link.active').id;
  // 탭에 따라 뉴스 소스 파라미터 설정
  let tabParam = (activeTabId === 'yahoo-tab') ? 'yahoo' : 'polygon';
  // 필터링된 뉴스 페이지로 이동
  window.location.href = `/news?name=${ticker}&source=${tabParam}`;
}

// 페이지 로드 완료 시 실행
document.addEventListener("DOMContentLoaded", function () {
  // URL 쿼리 스트링 가져오기
  const queryString = window.location.search;
  // 뉴스 데이터 API 호출
  fetch('/api/news' + queryString)
    .then(res => res.json())
    .then(data => {
      // Polygon과 Yahoo 뉴스 데이터 렌더링
      renderNews(data.polygon || [], 'polygon');
      renderNews(data.yahoo || [], 'yahoo');
    })
    .catch(() => {
      // 에러 발생 시 에러 메시지 표시
      document.getElementById('polygon-news').innerHTML = '<p>오류 발생</p>';
      document.getElementById('yahoo-news').innerHTML = '<p>오류 발생</p>';
    });

  // 뉴스 데이터를 화면에 렌더링하는 함수
  function renderNews(newsList, type) {
    const container = document.getElementById(`${type}-news`);
    const loadMoreBtn = document.getElementById(`${type}-load-more`);
    let currentIndex = 0;
    const perPage = 10; // 한 페이지당 표시할 뉴스 개수

    // 다음 페이지의 뉴스를 표시하는 함수
    function showNext() {
      const end = currentIndex + perPage;
      const pageItems = newsList.slice(currentIndex, end);
      
      // 데이터가 없는 경우 처리
      if (pageItems.length === 0 && currentIndex === 0) {
        container.innerHTML = '<p>데이터가 없습니다.</p>';
        loadMoreBtn.style.display = 'none';
        return;
      }

      // 각 뉴스 항목을 카드 형태로 표시
      pageItems.forEach(news => {
        const title = news.title || '(제목 없음)';
        const summary = news.summary || '';
        const sentiment = (news.sentiment || 'neutral').toLowerCase();
        const date = news.date || '-';
        const url = news.url || '#';

        // 감성 분석 결과에 따른 스타일 클래스 설정
        let sentimentClass = 'sentiment-neutral';
        if (sentiment === 'positive') sentimentClass = 'sentiment-positive';
        else if (sentiment === 'negative') sentimentClass = 'sentiment-negative';

        // 뉴스 카드 HTML 생성
        const div = document.createElement('div');
        div.className = 'news-card';
        div.innerHTML = `
          <h5><a href="${url}" target="_blank">${title}</a></h5>
          <p>${summary}</p>
          <div class="news-meta">
            <span class="sentiment-tag ${sentimentClass}">${sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}</span>
            <span>${date}</span>
            <a href="${url}" target="_blank"><i class="bi bi-link-45deg"></i> ${url}</a>
          </div>
        `;
        container.appendChild(div);
      });

      // 다음 페이지 인덱스 업데이트
      currentIndex += perPage;
      // 더 이상 표시할 뉴스가 없으면 '더 보기' 버튼 숨김
      if (currentIndex >= newsList.length) {
        loadMoreBtn.style.display = 'none';
      }
    }

    // '더 보기' 버튼 이벤트 리스너 설정 및 초기 데이터 표시
    if (loadMoreBtn) {
      loadMoreBtn.addEventListener('click', showNext);
      showNext();
    }
  }
});
