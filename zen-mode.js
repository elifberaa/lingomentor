// zen-mode.js

(function () {
    // --- KONFİGÜRASYON ---
    const MODEL_URL = 'models/tfjs_model/model.json'; // Modelin yolu
    const FATIGUE_THRESHOLD = 0.50; // Hassasiyet (0.0 ile 1.0 arası). Süre kontrolü olduğu için eşik daha da düşürüldü. 
    const REQUIRED_FATIGUE_TIME_MS = 3000; // Gözlerin en az kaç milisaniye kapalı kalması gerektiği (3000 ms = 3 Saniye)

    let fatigueStartTime = 0;
    let isZenModeActive = false;
    let mobilenetModel = null;
    let videoElement = null;

    // 1. TensorFlow.js kütüphanesini dinamik olarak yükle
    function loadTFJS() {
        return new Promise((resolve, reject) => {
            if (window.tf) {
                resolve();
                return;
            }
            const script = document.createElement('script');
            script.src = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs";
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    // 2. Gizli video elementini oluştur ve DOM'a ekle
    function createVideoElement() {
        videoElement = document.createElement('video');
        videoElement.id = 'zen-mode-webcam';
        videoElement.width = 224;
        videoElement.height = 224;
        videoElement.autoplay = true;
        videoElement.playsInline = true;
        videoElement.muted = true;

        // Kullanıcıyı rahatsız etmemek için videoyu gizliyoruz ama Safari'nin kamerayı dondurmaması için display:none YERİNE görünmez yapıyoruz.
        videoElement.style.position = 'absolute';
        videoElement.style.opacity = '0';
        videoElement.style.pointerEvents = 'none';
        videoElement.style.zIndex = '-1';
        videoElement.style.objectFit = 'cover'; // Kameranın yüzü basıklaştırmasını/bozmasını engellemek için
        document.body.appendChild(videoElement);
    }

    // 3. Kamerayı başlat ve Modeli yükle
    async function initZenMode() {
        try {
            console.log("[Zen Mode] Kamera izni isteniyor...");
            
            // Medya cihazları API'si desteklenmiyorsa (HTTP üzerinden bağlanıldığında olabilir)
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error("Tarayıcınız kamera erişimini desteklemiyor veya güvenli olmayan bir bağlantı (HTTP) kullanıyorsunuz.");
            }

            // Önce kamera iznini al
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoElement.srcObject = stream;

            await new Promise((resolve) => {
                videoElement.onloadedmetadata = () => {
                    videoElement.play(); // Safari'de akışın donmasını engellemek için manuel tetikleme
                    console.log("[Zen Mode] Kamera başlatıldı.");
                    resolve();
                };
            });

            console.log("[Zen Mode] TFJS modeli yükleniyor...");
            try {
                mobilenetModel = await window.tf.loadLayersModel(MODEL_URL);
                console.log("[Zen Mode] Model başarıyla yüklendi.");
            } catch (modelError) {
                console.error("[Zen Mode] Model yüklenemedi.", modelError);
                
                // Kullanıcıya sayfayı file:// ile açtığı için hata aldığını gösteren büyük bir uyarı ekleyelim
                const errorOverlay = document.createElement('div');
                Object.assign(errorOverlay.style, {
                    position: 'fixed', top: '0', left: '0', width: '100vw', height: '100vh',
                    backgroundColor: 'rgba(200, 0, 0, 0.9)', color: 'white', display: 'flex',
                    flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                    zIndex: '999999', fontFamily: 'sans-serif', textAlign: 'center', padding: '20px'
                });
                errorOverlay.innerHTML = `
                    <h1 style="font-size: 3rem; margin-bottom: 20px;">🚨 Yapay Zeka Modeli Yüklenemedi! 🚨</h1>
                    <p style="font-size: 1.5rem;">Bunun sebebi sayfayı doğrudan bir klasörden (<b>file://</b>) açmış olmanızdır.</p>
                    <p style="font-size: 1.5rem;">Tarayıcınız güvenlik sebebiyle model dosyalarının okunmasına izin vermiyor.</p>
                    <br>
                    <p style="font-size: 1.5rem; background: black; padding: 15px; border-radius: 10px; word-wrap: break-word; max-width: 80%;">
                        <b>Hata Detayı:</b> ${modelError.message || modelError}
                    </p>
                    <p style="font-size: 1.5rem; background: black; padding: 15px; border-radius: 10px;">
                        ÇÖZÜM: VS Code'da eklentilerden <b>Live Server</b> kurup projeyi onunla açmalısınız.
                    </p>
                `;
                document.body.appendChild(errorOverlay);
                
                return;
            }

            console.log("[Zen Mode] İzleme başladı.");
            
            // Kullanıcının sorunu görebilmesi için Hata ayıklama (Debug) arayüzü
            let debugUI = document.getElementById('zen-debug-ui');
            if(!debugUI) {
                debugUI = document.createElement('div');
                debugUI.id = 'zen-debug-ui';
                Object.assign(debugUI.style, {
                    position: 'fixed',
                    bottom: '10px',
                    right: '10px',
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    color: '#00ff00',
                    padding: '10px',
                    zIndex: '999999',
                    fontFamily: 'monospace',
                    fontSize: '14px',
                    borderRadius: '8px',
                    pointerEvents: 'none',
                    minWidth: '200px'
                });
                document.body.appendChild(debugUI);
            }

            detectLoop();
            
        } catch (error) {
            console.error("[Zen Mode] Başlatma hatası. Kamera izni reddedilmiş olabilir.", error);
        }
    }

    // 4. Tespit Döngüsü
    async function detectLoop() {
        if (mobilenetModel && videoElement.readyState === 4) {
            const predictionArray = window.tf.tidy(() => {
                let imgTensor = window.tf.browser.fromPixels(videoElement);
                imgTensor = window.tf.image.resizeBilinear(imgTensor, [224, 224]);

                // [-1, 1] Normalizasyonu (MobileNetV2 için)
                const normalizedTensor = imgTensor.div(window.tf.scalar(127.5)).sub(window.tf.scalar(1.0));
                const batchedTensor = normalizedTensor.expandDims(0);
                const prediction = mobilenetModel.predict(batchedTensor);
                return prediction.dataSync(); // Tüm diziyi döndür
            });

            // Ekranda canlı verileri göster
            let debugText = "<b>[Canlı Model Verileri]</b><br>";
            for(let i=0; i<predictionArray.length; i++) {
                debugText += `Sınıf ${i}: ${predictionArray[i].toFixed(4)}<br>`;
            }

            // Eğer Sınıf 1 işe yaramadıysa, belki Sınıf 0 kapalı gözdür. 
            // Şimdilik Index 1'i kullanıyoruz ama ekranda ikisini de göreceksiniz.
            const score0 = predictionArray[0];
            const score1 = predictionArray.length > 1 ? predictionArray[1] : 0;
            
            // Kullanılacak skoru Sınıf 0 veya Sınıf 1 olarak ayarla
            // ŞİMDİLİK SINIF 0'I BAZ ALALIM (Genelde yorgunluk/kapalı göz 0. indekste olur)
            const score = score0; 

            debugText += `<br>Sınıf 0: ${score0.toFixed(4)}`;
            if (predictionArray.length > 1) {
                debugText += `<br>Sınıf 1: ${score1.toFixed(4)}<br>`;
            } else {
                debugText += `<br>`;
            }
            
            // Zamana dayalı kontrol (Frame yerine)
            if (score >= FATIGUE_THRESHOLD) {
                if (fatigueStartTime === 0) {
                    fatigueStartTime = Date.now(); // Göz kapanmaya başladığı anı kaydet
                }
                
                const elapsedTime = Date.now() - fatigueStartTime;
                debugText += `Geçen Süre: ${(elapsedTime / 1000).toFixed(1)} saniye / ${(REQUIRED_FATIGUE_TIME_MS / 1000).toFixed(1)}s`;

                if (elapsedTime >= REQUIRED_FATIGUE_TIME_MS && !isZenModeActive) {
                    triggerZenModeAction();
                }
            } else {
                fatigueStartTime = 0; // Göz açıldığında süreyi sıfırla
                debugText += `Geçen Süre: 0.0 saniye / ${(REQUIRED_FATIGUE_TIME_MS / 1000).toFixed(1)}s`;
            }

            const debugEl = document.getElementById('zen-debug-ui');
            if (debugEl) {
                debugEl.innerHTML = debugText;
            }
        }

        // Döngüyü devam ettir
        requestAnimationFrame(detectLoop);
    }

    // Bildirim Sesini Çal (Web Audio API kullanarak yumuşak bir sinyal üretir)
    function playAlertSound() {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            const ctx = new AudioContext();
            
            // İlk nota (Ding)
            const osc1 = ctx.createOscillator();
            const gain1 = ctx.createGain();
            osc1.type = 'sine';
            osc1.frequency.setValueAtTime(523.25, ctx.currentTime); // C5 (Do)
            
            gain1.gain.setValueAtTime(0, ctx.currentTime);
            gain1.gain.linearRampToValueAtTime(0.5, ctx.currentTime + 0.05);
            gain1.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
            
            osc1.connect(gain1);
            gain1.connect(ctx.destination);
            osc1.start(ctx.currentTime);
            osc1.stop(ctx.currentTime + 0.5);

            // İkinci nota (Dong - biraz daha pes)
            const osc2 = ctx.createOscillator();
            const gain2 = ctx.createGain();
            osc2.type = 'sine';
            osc2.frequency.setValueAtTime(440, ctx.currentTime + 0.15); // A4 (La)
            
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

    // 5. Zen Modu Tetiklendiğinde Olacaklar (Ekrana Uyarı Çıkarma)
    function triggerZenModeAction() {
        isZenModeActive = true;
        console.log("🚨 Zen Modu: Yorgunluk/Odak kaybı tespit edildi!");
        
        playAlertSound(); // Uyarı sesini tetikle

        // Örnek bir görsel uyarı: Ekranı karartma ve mesaj gösterme
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

        // Uyarıyı 5 saniye sonra kaldır
        setTimeout(() => {
            const el = document.getElementById('zen-mode-overlay');
            if (el) el.remove();
            isZenModeActive = false;
            fatigueStartTime = 0;
            console.log("ℹ️ Zen Modu uyarısı kapandı, izlemeye devam ediliyor...");
        }, 5000);
    }

    // Sistemi ayağa kaldıran fonksiyon
    async function startZenMode() {
        createVideoElement();
        try {
            await loadTFJS();
            await initZenMode();
        } catch (error) {
            console.error("[Zen Mode] TFJS yükleme hatası:", error);
        }
    }

    // Sayfa DOM içeriği yüklendiğinde veya çoktan yüklendiyse sistemi ayağa kaldır
    if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', startZenMode);
    } else {
        startZenMode();
    }

})();
