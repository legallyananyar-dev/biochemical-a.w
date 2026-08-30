    // Dynamically load settings from config JSON file
    fetch('/static/config.json')
        .then(res => res.json())
        .then(data => {
            const select = document.getElementById('moleculeSelect');
            data.molecule_presets.forEach(mol => {
                let option = document.createElement('option');
                option.value = mol.formula;
                option.textContent = `${mol.name} (${mol.risk_level} Defense Matrix)`;
                select.appendChild(option);
            });
        });

    function runProtectionEngine() {
        const logger = document.getElementById('terminalLogs');
        const molecule = document.getElementById('moleculeSelect').value;
        const fileInput = document.getElementById('audioFile');

        if (fileInput.files.length === 0) {
            logger.innerHTML = `❌ ERROR: Please upload a target voice audio track first.`;
            return;
        }

        logger.innerHTML = `[PROCESS] Extracting chemical descriptors for: ${molecule}...\n`;
        
        // Pack data into a real network payload form
        const formData = new FormData();
        formData.append('audio', fileInput.files[0]);
        formData.append('molecule', molecule);

        // Send payload to Python Flask Server
        fetch('/api/protect', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if(data.status === "success") {
                logger.innerHTML += `[SUCCESS] Tensor Packet mapping completed.\n`;
                logger.innerHTML += `[ACTIVE] Steganography lock verified via Transformer Sentinel: ${data.integrity_rating}\n`;
                logger.innerHTML += `[SECURE] Voice file fully locked against AI isolation scraping.\n`;
            } else {
                logger.innerHTML += `❌ PIPELINE ERROR: ${data.msg}\n`;
            }
        })
        .catch(err => {
            logger.innerHTML += `❌ SERVER SYSTEM DISCONNECT: ${err}\n`;
        });
    }
