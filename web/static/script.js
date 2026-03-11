// Main JavaScript for Insurance Fraud Detection System

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('fraudDetectionForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');

    // Load models info
    loadModelsInfo();

    // Form submission
    form.addEventListener('submit', async function (e) {
        e.preventDefault();

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

function setLoadingState(isLoading) {
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');

    if (isLoading) {
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
        btnText.textContent = 'Đang phân tích...';
        submitBtn.querySelector('.btn-icon').innerHTML = '<span class="spinner"></span>';
    } else {
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
        btnText.textContent = 'Phân tích & Phát hiện Gian lận';
        submitBtn.querySelector('.btn-icon').textContent = '🔍';
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
        modelsGrid.appendChild(modelCard);
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

// Add tooltips for better UX
const tooltips = {
    'age': 'Tuổi của người yêu cầu bảo hiểm',
    'income': 'Thu nhập hàng tháng tính bằng VNĐ',
    'claim_amount': 'Số tiền yêu cầu bồi thường',
    'num_claims': 'Số lần đã yêu cầu bồi thường trước đây',
    'policy_duration': 'Số tháng đã tham gia bảo hiểm',
    'credit_score': 'Điểm tín dụng (300-850)'
};

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
