// zen-mode.js

(function () {
    // --- KONFİGÜRASYON ---
    const EAR_THRESHOLD = 0.20; // Göz kırpma/kapalı eşiği (Daha katı olması için 0.20'ye düşürüldü)
    const REQUIRED_FATIGUE_TIME_MS = 4000; // Gözlerin en az kaç milisaniye kapalı kalması gerektiği (4000 ms = 4 Saniye)

    let fatigueStartTime = 0;
    let isZenModeActive = false;
    let faceDetector = null;
    let videoElement = null;

    // 1. TensorFlow.js ve Face Landmarks kütüphanelerini dinamik yükle
    async function loadLibraries() {
        if (!window.tf) {
            await new Promise((resolve) => {
                const script = document.createElement('script');
                script.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs";
                script.onload = resolve;
                document.head.appendChild(script);
            });
        }
        if (!window.faceLandmarksDetection) {
            await new Promise((resolve) => {
                const script = document.createElement('script');
                script.src = "https://cdn.jsdelivr.net/npm/@tensorflow-models/face-landmarks-detection";
                script.onload = resolve;
                document.head.appendChild(script);
            });
        }
    }

    // 2. Gizli video elementini oluştur ve DOM'a ekle
    function createVideoElement() {
        videoElement = document.createElement('video');
        videoElement.id = 'zen-mode-webcam';
        videoElement.width = 640;
        videoElement.height = 480;
        videoElement.autoplay = true;
        videoElement.playsInline = true;
        videoElement.muted = true;

        videoElement.style.position = 'absolute';
        videoElement.style.opacity = '0';
        videoElement.style.pointerEvents = 'none';
        videoElement.style.zIndex = '-1';
        videoElement.style.objectFit = 'cover';
        document.body.appendChild(videoElement);
    }

    // 3. Kamerayı başlat ve Face Mesh modelini yükle
    async function initZenMode() {
        try {
            console.log("[Zen Mode] Yüz Tanıma (Face Mesh) kütüphaneleri yükleniyor...");
            await loadLibraries();
            
            createVideoElement();

            console.log("[Zen Mode] Kamera izni isteniyor...");
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: 640, height: 480 }
            });
            videoElement.srcObject = stream;

            await new Promise((resolve) => {
                videoElement.onloadedmetadata = () => {
                    videoElement.play(); 
                    console.log("[Zen Mode] Kamera başlatıldı.");
                    resolve();
                };
            });

            console.log("[Zen Mode] Face Mesh Modeli yükleniyor... (Bu işlem birkaç saniye sürebilir)");
            const model = window.faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
            const detectorConfig = {
                runtime: 'tfjs', // Tarayıcı hızlandırması için WebGL kullanır
                refineLandmarks: true // Göz bebeklerini ve göz çevresini daha net algılar
            };
            faceDetector = await window.faceLandmarksDetection.createDetector(model, detectorConfig);
            console.log("[Zen Mode] Kusursuz Face Mesh Modeli başarıyla yüklendi! Tespit başlıyor.");

            createDebugUI();
            requestAnimationFrame(detectLoop);

        } catch (error) {
            console.error("[Zen Mode] Başlatma Hatası:", error);
            
            // Hata arayüzü
            const errorOverlay = document.createElement('div');
            Object.assign(errorOverlay.style, {
                position: 'fixed', top: '0', left: '0', width: '100vw', height: '100vh',
                backgroundColor: 'rgba(255,0,0,0.9)', color: 'white', zIndex: '999999',
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                fontFamily: 'sans-serif', textAlign: 'center', padding: '20px'
            });
            errorOverlay.innerHTML = `
                <h1 style="font-size: 2rem;">🚨 Zen Modu Başlatılamadı!</h1>
                <p style="font-size: 1.2rem; margin-top:10px;">Kamera izni verilmemiş olabilir veya model yüklenemedi.</p>
                <div style="background:rgba(0,0,0,0.5); padding:15px; margin-top:20px; border-radius:8px; font-family:monospace;">
                    Hata Detayı: ${error.message || error}
                </div>
                <button onclick="this.parentElement.remove()" style="margin-top:20px; padding:10px 20px; font-size:1rem; cursor:pointer;">Kapat</button>
            `;
            document.body.appendChild(errorOverlay);
        }
    }

    // Öklid Uzaklığı (İki nokta arası mesafe)
    function distance(p1, p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

    // Göz Açıklık Oranı (EAR - Eye Aspect Ratio) Hesaplama
    function calculateEAR(landmarks, eyeIndices) {
        // eyeIndices sırası: [DışKöşe, Üst1, Üst2, İçKöşe, Alt2, Alt1]
        const p0 = landmarks[eyeIndices[0]]; 
        const p1 = landmarks[eyeIndices[1]]; 
        const p2 = landmarks[eyeIndices[2]]; 
        const p3 = landmarks[eyeIndices[3]]; 
        const p4 = landmarks[eyeIndices[4]]; 
        const p5 = landmarks[eyeIndices[5]]; 

        const vertical1 = distance(p1, p5);
        const vertical2 = distance(p2, p4);
        const horizontal = distance(p0, p3);

        if (horizontal === 0) return 0;
        return (vertical1 + vertical2) / (2.0 * horizontal);
    }

    // 4. Tespit Döngüsü (Her karede milimetrik tarama yapar)
    async function detectLoop() {
        if (!isZenModeActive && videoElement.readyState >= 2) {
            
            // Yüzü algıla
            const faces = await faceDetector.estimateFaces(videoElement);
            
            let debugText = "<b>Yüz Algılama (FaceMesh):</b> Çalışıyor 🟢<br>";
            
            if (faces.length > 0) {
                const keypoints = faces[0].keypoints;
                
                // Sağ Göz Noktaları: 33 (Dış), 160(Üst), 158(Üst), 133(İç), 153(Alt), 144(Alt)
                const rightEyeEAR = calculateEAR(keypoints, [33, 160, 158, 133, 153, 144]);
                
                // Sol Göz Noktaları: 362(İç), 385(Üst), 387(Üst), 263(Dış), 373(Alt), 380(Alt)
                const leftEyeEAR = calculateEAR(keypoints, [362, 385, 387, 263, 373, 380]);
                
                // İki gözün ortalama açıklık oranı (EAR)
                const avgEAR = (rightEyeEAR + leftEyeEAR) / 2;

                debugText += `<br>Göz Açıklığı (EAR): <b>${avgEAR.toFixed(3)}</b><br>`;
                debugText += `(Kapanma Sınırı: ${EAR_THRESHOLD.toFixed(3)})<br><br>`;
                
                // EAR eşiğin ALTINDAYSA göz kapalı (kısık) demektir
                if (avgEAR < EAR_THRESHOLD) {
                    if (fatigueStartTime === 0) {
                        fatigueStartTime = Date.now();
                    }
                    const elapsedTime = Date.now() - fatigueStartTime;
                    debugText += `<span style="color:#ff4444">⏳ Göz Kapalı Süre: ${(elapsedTime / 1000).toFixed(1)} saniye / ${(REQUIRED_FATIGUE_TIME_MS / 1000).toFixed(1)}s</span>`;

                    if (elapsedTime >= REQUIRED_FATIGUE_TIME_MS && !isZenModeActive) {
                        triggerZenModeAction();
                    }
                } else {
                    fatigueStartTime = 0;
                    debugText += `<span style="color:#44ff44">Gözler Açık 👀</span><br>`;
                    debugText += `Geçen Süre: 0.0 saniye / ${(REQUIRED_FATIGUE_TIME_MS / 1000).toFixed(1)}s`;
                }
            } else {
                debugText += `<br><span style="color:orange">Yüz Algılanamadı! Kameraya bakın.</span>`;
                fatigueStartTime = 0; // Yüz yoksa süreyi sıfırla
            }

            const debugEl = document.getElementById('zen-debug-ui');
            if (debugEl) {
                debugEl.innerHTML = debugText;
            }
        }

        // Döngüyü çok yormamak için bir sonraki kareyi bekle
        requestAnimationFrame(detectLoop);
    }

    // Bildirim Sesini Çal (Web Audio API)
    function playAlertSound() {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            const ctx = new AudioContext();
            
            const osc1 = ctx.createOscillator();
            const gain1 = ctx.createGain();
            osc1.type = 'sine';
            osc1.frequency.setValueAtTime(523.25, ctx.currentTime);
            
            gain1.gain.setValueAtTime(0, ctx.currentTime);
            gain1.gain.linearRampToValueAtTime(0.5, ctx.currentTime + 0.05);
            gain1.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
            
            osc1.connect(gain1);
            gain1.connect(ctx.destination);
            osc1.start(ctx.currentTime);
            osc1.stop(ctx.currentTime + 0.5);

            const osc2 = ctx.createOscillator();
            const gain2 = ctx.createGain();
            osc2.type = 'sine';
            osc2.frequency.setValueAtTime(440, ctx.currentTime + 0.15);
            
            gain2.gain.setValueAtTime(0, ctx.currentTime + 0.15);
            gain2.gain.linearRampToValueAtTime(0.5, ctx.currentTime + 0.2);
            gain2.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.8);
            
            osc2.connect(gain2);
            gain2.connect(ctx.destination);
            osc2.start(ctx.currentTime + 0.15);
            osc2.stop(ctx.currentTime + 0.8);

        } catch (e) {
            console.warn("Ses çalınamadı: ", e);
        }
    }

    // 5. Zen Modu Tetiklendiğinde Olacaklar
    function triggerZenModeAction() {
        isZenModeActive = true;
        console.log("🚨 Zen Modu: Yorgunluk/Odak kaybı tespit edildi!");
        
        playAlertSound();

        const overlay = document.createElement('div');
        overlay.id = 'zen-mode-overlay';
        Object.assign(overlay.style, {
            position: 'fixed',
            top: '0',
            left: '0',
            width: '100vw',
            height: '100vh',
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            color: 'white',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: '999999',
            fontFamily: 'sans-serif',
            textAlign: 'center'
        });

        overlay.innerHTML = `
            <h1 style="font-size: 3rem; margin-bottom: 20px;">Gözlerin Yorulmuş Görünüyor 😴</h1>
            <p style="font-size: 1.5rem;">Odaklanma seviyen düştü. Biraz ara vermeye ne dersin?</p>
        `;
        document.body.appendChild(overlay);

        setTimeout(() => {
            const el = document.getElementById('zen-mode-overlay');
            if (el) el.remove();
            isZenModeActive = false;
            fatigueStartTime = 0;
            console.log("ℹ️ Zen Modu uyarısı kapandı, izlemeye devam ediliyor...");
        }, 5000);
    }

    // Hata Ayıklama UI Oluşturucu
    function createDebugUI() {
        let debugEl = document.getElementById('zen-debug-ui');
        if (!debugEl) {
            debugEl = document.createElement('div');
            debugEl.id = 'zen-debug-ui';
            Object.assign(debugEl.style, {
                position: 'fixed',
                bottom: '20px',
                right: '20px',
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                color: '#0f0',
                padding: '10px 15px',
                borderRadius: '8px',
                fontFamily: 'monospace',
                fontSize: '14px',
                zIndex: '999999',
                pointerEvents: 'none'
            });
            document.body.appendChild(debugEl);
        }
    }

    // DOM yüklendiğinde kamerayı ve modeli başlat
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initZenMode);
    } else {
        initZenMode();
    }
})();
