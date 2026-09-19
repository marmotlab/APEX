document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.navbar-burger').forEach(burger => {
    burger.addEventListener('click', () => {
      const menu = burger.closest('.navbar').querySelector('.navbar-menu');
      const expanded = burger.classList.toggle('is-active');
      burger.setAttribute('aria-expanded', String(expanded));
      menu.classList.toggle('is-active', expanded);
    });
  });

  if (window.bulmaCarousel) {
    bulmaCarousel.attach('.carousel', {
      slidesToScroll: 1,
      slidesToShow: 3,
      loop: true,
      infinite: false,
      autoplay: false
    });
  }

  // Keep native controls usable if IntersectionObserver is unavailable.
  if (!('IntersectionObserver' in window)) return;

  const videos = document.querySelectorAll('video[data-autoplay]');
  const visibleVideos = new Set();
  const pausedByPage = new WeakSet();
  const pausedByUser = new WeakSet();
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const connection = navigator.connection;
  const canAutoplay = () => !reducedMotion.matches && !connection?.saveData;

  function pauseVideo(video) {
    if (!video.paused) {
      pausedByPage.add(video);
      video.pause();
    }
  }

  function playVideo(video) {
    if (document.hidden || !canAutoplay() || pausedByUser.has(video)) return;
    // Autoplay can be refused by the browser; the poster and controls remain.
    video.play().catch(() => {});
  }

  videos.forEach(video => {
    video.addEventListener('pause', () => {
      if (!pausedByPage.delete(video) && !video.ended) pausedByUser.add(video);
    });
    video.addEventListener('play', () => pausedByUser.delete(video));
  });

  const observer = new IntersectionObserver(entries => {
    entries.forEach(({ target: video, isIntersecting, intersectionRatio }) => {
      if (isIntersecting && intersectionRatio >= 0.1) {
        visibleVideos.add(video);
        playVideo(video);
      } else {
        visibleVideos.delete(video);
        pauseVideo(video);
      }
    });
  }, { threshold: [0, 0.1] });

  videos.forEach(video => observer.observe(video));

  function updatePlayback() {
    videos.forEach(video => {
      if (document.hidden || !canAutoplay() || !visibleVideos.has(video)) {
        pauseVideo(video);
      } else {
        playVideo(video);
      }
    });
  }

  document.addEventListener('visibilitychange', updatePlayback);
  reducedMotion.addEventListener('change', updatePlayback);
  connection?.addEventListener('change', updatePlayback);
});
