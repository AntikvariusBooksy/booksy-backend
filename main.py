<script>
    const BOOKSY_API_URL = "https://booksy-backend-production-4ee2.up.railway.app";
    
    // Globális konfiguráció, amit a szervertől kapunk
    let booksyConfig = {
        ui_lang: "hu", // Alapértelmezett
        bubble_text: "Booksy vagyok, az AI könyvtáros.", 
        placeholder: "Kérdezz bármit a könyvekről..."
    };

    function getSessionId() {
        let sessionId = localStorage.getItem('booksy_session_id');
        if (!sessionId) {
            sessionId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
            localStorage.setItem('booksy_session_id', sessionId);
        }
        return sessionId;
    }

    function saveChatHistory() {
        const msgs = document.getElementById("booksy-messages");
        if(msgs) localStorage.setItem("booksy_history_v1", msgs.innerHTML);
    }

    function loadChatHistory() {
        const history = localStorage.getItem("booksy_history_v1");
        const msgs = document.getElementById("booksy-messages");
        if (history && history.trim() !== "") {
            msgs.innerHTML = history;
            setTimeout(() => msgs.scrollTop = msgs.scrollHeight, 100);
        } else {
            showWelcomeMessage();
        }
    }

    function clearBooksyHistory() {
        const confirmText = booksyConfig.ui_lang === 'hu' ? 
            "Biztosan új beszélgetést szeretnél kezdeni?" : 
            "Sigur vrei să începi o conversație nouă?";
            
        if(confirm(confirmText)) {
            localStorage.removeItem("booksy_history_v1");
            document.getElementById("booksy-messages").innerHTML = "";
            showWelcomeMessage();
        }
    }

    function showWelcomeMessage() {
        // Dinamikus üdvözlés a szerver válasza alapján
        const logoHtml = `<img src="https://www.antikvarius.ro/wp-content/uploads/logo_transparent.svg" style="height:18px; width:auto; vertical-align:middle; margin: 0 2px 4px 0;" alt="Booksy">`;
        
        let greeting = "";
        let sub = "";
        
        if (booksyConfig.ui_lang === 'hu') {
            greeting = `Szia! ${logoHtml} ${booksyConfig.bubble_text || "Booksy vagyok."}`;
            sub = "(Okos Antikvár Asszisztens)";
        } else {
            greeting = `Bună! ${logoHtml} ${booksyConfig.bubble_text || "Sunt Booksy."}`;
            sub = "(Asistent Smart Anticariat)";
        }

        const msgDiv = document.createElement("div");
        msgDiv.className = "booksy-msg bot-msg";
        msgDiv.innerHTML = `${greeting}<br><i style="font-size:12px; color:#888; margin-top:5px; display:block;">${sub}</i>`;
        
        document.getElementById("booksy-messages").appendChild(msgDiv);
        
        const input = document.getElementById("booksy-input");
        if(input) input.placeholder = booksyConfig.placeholder;
    }
    
    // --- SMART HANDSHAKE ---
    async function initBooksy() {
        try {
            // "Kézfogás" az AI-val a háttérben
            const res = await fetch(`${BOOKSY_API_URL}/init-chat`, {
                method: "POST", 
                headers: {"Content-Type":"application/json"},
                body: JSON.stringify({ 
                    url: window.location.href, 
                    session_id: getSessionId() 
                })
            });
            if (res.ok) {
                const data = await res.json();
                // Felülírjuk az alapértelmezett configot az AI válaszával
                booksyConfig = data;
                console.log("Booksy Handshake Success:", data);
            }
        } catch (e) {
            console.warn("Booksy Handshake Failed (Offline mode):", e);
        }

        // UI Frissítése a kapott adatokkal
        const bubble = document.getElementById("booksy-proactive-bubble");
        if(bubble) bubble.innerText = booksyConfig.bubble_text;
        
        const input = document.getElementById("booksy-input");
        if(input) input.placeholder = booksyConfig.placeholder;

        const resetBtn = document.getElementById("booksy-reset-btn");
        if(resetBtn) resetBtn.title = booksyConfig.ui_lang === 'hu' ? "Új beszélgetés" : "Conversație nouă";
    }

    // Indítás késleltetve (de a handshake már megy közben)
    window.addEventListener('load', function() {
        loadChatHistory();
        
        // Azonnal indítjuk a handshake-et
        initBooksy().then(() => {
            // Ha kész a handshake, várunk a 8. másodpercig a megjelenéssel
            setTimeout(() => {
                const wrapper = document.getElementById("booksy-launcher-wrapper");
                if(wrapper) {
                    wrapper.classList.add("visible");
                    setTimeout(() => { wrapper.classList.add("wiggle"); }, 3000);
                    
                    const bubble = document.getElementById("booksy-proactive-bubble");
                    if(bubble) {
                        setTimeout(() => {
                            bubble.style.opacity = "0";
                            setTimeout(() => { bubble.style.display = "none"; }, 2000); 
                        }, 15000); 
                    }
                }
            }, 8000); 
        });

        document.addEventListener('click', function(event) {
            const chat = document.getElementById("booksy-chat-container");
            const toggleBtn = document.getElementById("booksy-toggle-btn");
            const bubble = document.getElementById("booksy-proactive-bubble");

            if (chat && chat.style.display === "flex") {
                if (!chat.contains(event.target) && !toggleBtn.contains(event.target) && !bubble.contains(event.target)) {
                    toggleBooksyChat(); 
                }
            }
        });
    });

    function toggleBooksyChat() {
        const chat = document.getElementById("booksy-chat-container");
        const bubble = document.getElementById("booksy-proactive-bubble");
        
        if (chat.style.display === "flex") { 
            chat.style.display = "none";
            document.body.style.overflow = ""; 
            if(bubble && bubble.style.opacity !== "0") { bubble.style.display = "block"; }
        } 
        else { 
            chat.style.display = "flex"; 
            document.body.style.overflow = "hidden"; 
            if(bubble) bubble.style.display = "none"; 
            setTimeout(() => {
                document.getElementById("booksy-input").focus();
                const msgs = document.getElementById("booksy-messages");
                msgs.scrollTop = msgs.scrollHeight;
            }, 100);
        }
    }
    
    function handleBooksyKeyPress(e) { if (e.key === "Enter") sendBooksyMessage(); }
    
    async function sendBooksyMessage() {
        const input = document.getElementById("booksy-input");
        const text = input.value.trim();
        if (!text) return;
        
        const msgs = document.getElementById("booksy-messages");
        msgs.innerHTML += `<div class="booksy-msg user-msg">${text}</div>`;
        saveChatHistory(); 
        
        input.value = "";
        msgs.scrollTop = msgs.scrollHeight;
        
        // UI Nyelv használata a Configból
        const uiLang = booksyConfig.ui_lang;
        
        const loadingText = uiLang === "ro" ? "Booksy scrie" : "Booksy ír";
        const btnText = uiLang === "ro" ? "Vezi oferta ➤" : "Megnézem ➤";
        const placeholderUrl = `https://placehold.co/150x190/eeeeee/333333?text=Booksy`;
        
        const loader = document.createElement("div");
        loader.className = "booksy-msg bot-msg loading-dots";
        loader.innerHTML = `<span>${loadingText}</span> <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="#555" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin:0 2px; vertical-align: middle;"><path d="M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5z"></path><line x1="16" y1="8" x2="2" y2="22"></line><line x1="17.5" y1="15" x2="9" y2="15"></line></svg>`;
        msgs.appendChild(loader);
        msgs.scrollTop = msgs.scrollHeight;

        try {
            const res = await fetch(`${BOOKSY_API_URL}/chat`, {
                method: "POST", 
                headers: {"Content-Type":"application/json"},
                body: JSON.stringify({ 
                    message: text, 
                    session_id: getSessionId(),
                    context_url: window.location.href 
                })
            });

            if (!res.ok) throw new Error("API Error");

            const data = await res.json();
            loader.remove();

            let formattedText = data.reply
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/- (.*?)(<br>|$)/g, '<li>$1</li>')
                .replace(/<li>/g, '<ul><li>').replace(/<\/li>(?!<li>)/g, '</li></ul>')
                .replace(/<\/ul><br><ul>/g, ''); 

            msgs.innerHTML += `<div class="booksy-msg bot-msg">${formattedText}</div>`;

            if (data.products && data.products.length > 0) {
                let cardsHtml = '<div class="product-carousel">';
                data.products.forEach(p => {
                    const img = (p.image && p.image !== "") ? p.image : placeholderUrl;
                    let priceDisplay = p.price ? p.price : (uiLang === "ro" ? "Vezi pe site" : "Nézd az oldalon");
                    cardsHtml += `
                    <div class="product-card">
                        <img src="${img}" class="product-img" alt="${p.title}" onerror="this.src='${placeholderUrl}'">
                        <div class="product-title" title="${p.title}">${p.title}</div>
                        <div class="product-price">${priceDisplay}</div>
                        <a href="${p.url}" target="_blank" class="product-btn">${btnText}</a>
                    </div>`;
                });
                cardsHtml += '</div>';
                msgs.innerHTML += cardsHtml;
            }
            saveChatHistory();
            msgs.scrollTop = msgs.scrollHeight;

        } catch (e) {
            console.error(e);
            loader.remove();
            const errorMsg = uiLang === "ro" ? "Eroare de conexiune." : "Szerver hiba. Kérlek próbáld később.";
            msgs.innerHTML += `<div class="booksy-msg bot-msg" style="color:#e74c3c">⚠️ ${errorMsg}</div>`;
            saveChatHistory();
        }
    }
</script>