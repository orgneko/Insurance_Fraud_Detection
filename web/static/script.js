// Main JavaScript for Insurance Fraud Detection System
// Giữ nguyên toàn bộ logic cũ, chỉ thêm validation và cải tiến UX

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('fraudDetectionForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');

    // Load models info
    loadModelsInfo();

    // Thêm validation real-time
    setupValidation();

    // Form submission - giữ nguyên logic gốc, thêm validation trước khi gửi
    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        // Validate form trước khi gửi
        if (!validateForm()) {
            return;
        }

        // Show loading state
        setLoadingState(true);

        // Collect form data
        const formData = {
            age: document.getElementById('age').value,
            income: document.getElementById('income').value,
            claim_amount: document.getElementById('claim_amount').value,
            num_claims: document.getElementById('num_claims').value,
            policy_duration: document.getElementById('policy_duration').value,
            num_dependents: document.getElementById('num_dependents').value,
            vehicle_age: document.getElementById('vehicle_age').value,
            credit_score: document.getElementById('credit_score').value,
            employment_status: document.getElementById('employment_status').value,
            education: document.getElementById('education').value,
            marital_status: document.getElementById('marital_status').value,
            claim_type: document.getElementById('claim_type').value
        };

        try {
            // Send prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            console.log("DEBUG RESULT:", result);

            if (result.success) {
                displayResults(result);

                // Scroll to results
                setTimeout(() => {
                    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 300);
            } else {
                alert('Lỗi: ' + result.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Có lỗi xảy ra khi phân tích. Vui lòng thử lại!');
        } finally {
            setLoadingState(false);
        }
    });
});

// ===== CÁC HÀM GỐC (GIỮ NGUYÊN) =====
function setLoadingState(isLoading) {
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnIcon = submitBtn.querySelector('.btn-icon');

    if (isLoading) {
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
        btnText.textContent = 'Đang phân tích...';
        if (btnIcon) btnIcon.style.display = 'none';
    } else {
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
        btnText.textContent = 'Phân tích & Phát hiện Gian lận';
        if (btnIcon) btnIcon.style.display = 'inline-flex';
    }
}

function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const finalResult = document.getElementById('finalResult');
    const modelsGrid = document.getElementById('modelsGrid');
    const analysisDetails = document.getElementById('analysisDetails');

    // Show results section
    resultsSection.style.display = 'block';

    // Display final result
    const isFraud = result.final_prediction === 1;
    const probability = (result.final_probability * 100).toFixed(2);

    finalResult.className = 'final-result ' + (isFraud ? 'fraud' : 'safe');
    finalResult.innerHTML = `
        <div class="result-icon">${isFraud ? '⚠️' : '✅'}</div>
        <h3 class="result-title">${isFraud ? 'CẢNH BÁO GIAN LẬN' : 'AN TOÀN'}</h3>
        <p class="result-subtitle">
            ${isFraud ?
            'Yêu cầu bảo hiểm này có dấu hiệu gian lận' :
            'Yêu cầu bảo hiểm này có vẻ hợp lệ'}
        </p>
        <div class="result-probability">${probability}%</div>
        <p class="result-subtitle">Xác suất gian lận</p>
    `;

    // Display individual model predictions
    modelsGrid.innerHTML = '';

    const modelNames = {
        'xgboost': 'XGBoost',
        'random_forest': 'Random Forest',
        'ann': 'ANN (Neural Network)'
    };

    for (const [modelKey, modelData] of Object.entries(result.predictions)) {
        const modelIsFraud = modelData.prediction === 1;
        const modelProb = (modelData.probability * 100).toFixed(2);

        const modelCard = document.createElement('div');
        modelCard.className = 'model-card';
        modelCard.innerHTML = `
            <div class="model-header">
                <h4 class="model-name">${modelNames[modelKey]}</h4>
                <span class="model-badge ${modelIsFraud ? 'fraud' : 'safe'}">
                    ${modelIsFraud ? 'FRAUD' : 'SAFE'}
                </span>
            </div>
            <div class="model-probability">${modelProb}%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${modelProb}%"></div>
            </div>
        `;
        modelCard.style.cursor = "pointer";

       const currentModel = modelKey;
        modelCard.addEventListener('click', () => {
            console.log("CLICK:", currentModel);
            console.log("SHAP:", result.shap[currentModel]);
            showModelExplain(currentModel, result.shap);
        });

        modelsGrid.appendChild(modelCard);
    }
    
    // 🔥 PHẦN QUAN TRỌNG: HIỂN THỊ LÝ DO
    let reasonsHTML = '';
   if (result.reasons && result.reasons.length > 0) {
    reasonsHTML = `
        <div class="detail-item">
            <span class="detail-label">Lý do nghi ngờ:</span>
            <ul class="reason-list">
                ${result.reasons.map(r => `<li>${r}</li>`).join('')}
            </ul>
        </div>
    `;
} else {
    reasonsHTML = `
        <div class="detail-item">
            <span class="detail-label">Lý do:</span>
            <span class="detail-value">Không phát hiện dấu hiệu bất thường</span>
        </div>
    `;
}
    // Display analysis details
    analysisDetails.innerHTML = `
        <div class="detail-item">
            <span class="detail-label">Số mô hình phát hiện gian lận:</span>
            <span class="detail-value">${result.fraud_votes}/${result.total_models}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Xác suất trung bình:</span>
            <span class="detail-value">${probability}%</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Kết luận:</span>
            <span class="detail-value" style="color: ${isFraud ? 'var(--danger)' : 'var(--success)'}">
                ${isFraud ? 'CẦN KIỂM TRA KỸ' : 'CÓ THỂ CHẤP NHẬN'}
            </span>
        </div>
        ${reasonsHTML}  
    `;

    // Animate confidence bars
    setTimeout(() => {
        document.querySelectorAll('.confidence-fill').forEach(fill => {
            fill.style.width = fill.style.width;
        });
    }, 100);
}

async function loadModelsInfo() {
    try {
        const response = await fetch('/models-info');
        const info = await response.json();

        if (info.total_models > 0) {
            document.getElementById('modelsCount').textContent = info.total_models;
        }
    } catch (error) {
        console.error('Error loading models info:', error);
    }
}

function showModelExplain(modelName, shapData) {
    const data = shapData[modelName];

    if (!data || data.length === 0) {
        showPopup(`
            <div class="shap-container">
                <h3>⚠️ ${modelName.toUpperCase()}</h3>
                <p>Model này chưa hỗ trợ giải thích (SHAP)</p>
            </div>
        `);
        return;
    }

    // Map tên tiếng Việt
    const featureMap = {
        claim_amount: "Số tiền yêu cầu",
        claim_to_income_ratio: "Tỷ lệ yêu cầu / thu nhập",
        credit_score: "Điểm tín dụng",
        income: "Thu nhập",
        policy_duration: "Thời gian tham gia bảo hiểm"
    };

    // Sắp xếp theo mức độ ảnh hưởng mạnh nhất
    data.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

    let html = `
        <div class="shap-container">
            <h3>🔍 Giải thích mô hình: ${modelName.toUpperCase()}</h3>
            <p style="font-size:13px;color:#666;margin-bottom:15px;">
                Thanh đỏ: ▲ tăng nguy cơ gian lận | Thanh xanh: ▼ giảm nguy cơ gian lận
            </p>
    `;

    data.forEach(item => {
        const impact = item.impact;
        const percent = Math.min(Math.abs(impact) * 100, 100).toFixed(1);

        const isIncrease = impact > 0;

        const label = isIncrease
            ? "🔺 Tăng nguy cơ gian lận"
            : "🔻 Giảm nguy cơ gian lận";

        const colorClass = isIncrease ? "positive" : "negative";

        const featureName = featureMap[item.feature] || item.feature;

        html += `
            <div class="shap-row">
                <div class="shap-feature">${featureName}</div>

                <div class="shap-bar">
                    <div class="shap-fill ${colorClass}"
                         style="width:${percent}%">
                    </div>
                </div>

                <div class="shap-value ${colorClass}">
                    ${label}
                </div>
            </div>
        `;
    });

    html += `</div>`;

    showPopup(html);
}

function showPopup(content) {
    let popup = document.getElementById('popupExplain');

    if (!popup) {
        popup = document.createElement('div');
        popup.id = 'popupExplain';
        document.body.appendChild(popup);
    }

    popup.innerHTML = content + `
        <div style="text-align:center;margin-top:20px;">
            <button onclick="document.getElementById('popupExplain').remove()"
                style="padding:10px 24px;border:none;background:var(--primary);color:white;border-radius:40px;cursor:pointer;font-weight:600;">
                Đóng
            </button>
        </div>
    `;
}

// Format currency inputs
document.addEventListener('DOMContentLoaded', function () {
    const currencyInputs = ['income', 'claim_amount'];

    currencyInputs.forEach(id => {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener('blur', function () {
                const value = parseInt(this.value);
                if (!isNaN(value)) {
                    this.value = value;
                }
            });
        }
    });
});

// Add input validation
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('fraudDetectionForm');
    const inputs = form.querySelectorAll('input[type="number"]');

    inputs.forEach(input => {
        input.addEventListener('input', function () {
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            const value = parseFloat(this.value);

            if (!isNaN(min) && value < min) {
                this.value = min;
            }
            if (!isNaN(max) && value > max) {
                this.value = max;
            }
        });
    });
});

// ===== HÀM MỚI: VALIDATION FORM =====
function setupValidation() {
    const fields = [
        { id: 'age', min: 18, max: 80, errorId: 'ageError', message: 'Tuổi phải từ 18 đến 80' },
        { id: 'income', min: 0, errorId: 'incomeError', message: 'Thu nhập phải lớn hơn 0' },
        { id: 'claim_amount', min: 0, errorId: 'claimError', message: 'Số tiền bồi thường phải lớn hơn 0' },
        { id: 'num_claims', min: 0, max: 20, errorId: 'claimsError', message: 'Số lần yêu cầu từ 0 đến 20' },
        { id: 'policy_duration', min: 1, max: 120, errorId: 'durationError', message: 'Thời gian tham gia từ 1 đến 120 tháng' },
        { id: 'num_dependents', min: 0, max: 10, errorId: 'dependentsError', message: 'Số người phụ thuộc từ 0 đến 10' },
        { id: 'vehicle_age', min: 0, max: 30, errorId: 'vehicleError', message: 'Tuổi xe từ 0 đến 30 năm' },
        { id: 'credit_score', min: 300, max: 850, errorId: 'creditError', message: 'Điểm tín dụng từ 300 đến 850' }
    ];

    fields.forEach(field => {
        const input = document.getElementById(field.id);
        if (input) {
            input.addEventListener('input', () => validateField(field));
            input.addEventListener('blur', () => validateField(field));
        }
    });
}

function validateField(field) {
    const input = document.getElementById(field.id);
    const errorSpan = document.getElementById(field.errorId);
    const value = parseFloat(input.value);
    const parent = input.closest('.form-group');

    let isValid = true;
    let errorMsg = '';

    if (isNaN(value)) {
        isValid = false;
        errorMsg = 'Vui lòng nhập số hợp lệ';
    } else if (field.min !== undefined && value < field.min) {
        isValid = false;
        errorMsg = field.message || `Giá trị phải >= ${field.min}`;
    } else if (field.max !== undefined && value > field.max) {
        isValid = false;
        errorMsg = field.message || `Giá trị phải <= ${field.max}`;
    }

    if (!isValid) {
        parent.classList.add('error');
        if (errorSpan) errorSpan.textContent = errorMsg;
    } else {
        parent.classList.remove('error');
        if (errorSpan) errorSpan.textContent = '';
    }

    return isValid;
}

function validateForm() {
    const fields = [
        { id: 'age', min: 18, max: 80 },
        { id: 'income', min: 0 },
        { id: 'claim_amount', min: 0 },
        { id: 'num_claims', min: 0, max: 20 },
        { id: 'policy_duration', min: 1, max: 120 },
        { id: 'num_dependents', min: 0, max: 10 },
        { id: 'vehicle_age', min: 0, max: 30 },
        { id: 'credit_score', min: 300, max: 850 }
    ];

    let isValid = true;
    fields.forEach(field => {
        if (!validateField(field)) isValid = false;
    });

    // Kiểm tra các trường select không được để trống (mặc định đã có value)
    const selects = ['education', 'employment_status', 'marital_status', 'claim_type'];
    selects.forEach(id => {
        const select = document.getElementById(id);
        if (select && !select.value) {
            isValid = false;
            select.style.borderColor = 'var(--danger)';
            setTimeout(() => { select.style.borderColor = ''; }, 2000);
        }
    });

    if (!isValid) {
        alert('Vui lòng kiểm tra lại thông tin nhập vào!');
    }

    return isValid;
}