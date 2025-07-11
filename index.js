document.getElementById('batchForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const applicantName = document.getElementById('applicantName').value.trim();
    const batchResultDiv = document.getElementById('batchResult');
    batchResultDiv.innerHTML = 'Processing...';

    if (!applicantName) {
        batchResultDiv.innerHTML = '<span style="color:red">Please enter applicant name.</span>';
        return;
    }
    if (selectedFiles.length === 0) {
        batchResultDiv.innerHTML = '<span style="color:red">Please select at least one file.</span>';
        return;
    }

    const formData = new FormData();
    formData.append('applicant_name', applicantName);
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });

    try {
        const response = await fetch('/verify_multiple', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) {
            batchResultDiv.innerHTML = `<span style='color:red'>${data.error || 'Batch verification failed.'}</span>`;
            return;
        }
        let html = `<b>Applicant Name:</b> ${data.applicant_name}<br><b>Results:</b><ul>`;
        for (const doc of data.results) {
            html += `<li><b>${doc.filename}</b> - <b>Type:</b> ${doc.document_type}`;
            if (doc.classification_score !== undefined) {
                html += ` <span style='color:#888'>(score: ${doc.classification_score})</span>`;
            }
            html += '<ul>';
            for (const [key, value] of Object.entries(doc.fields || {})) {
                html += `<li><b>${key}:</b> ${value || '<i>not found</i>'}</li>`;
            }
            html += '</ul>';
            if (doc.annotated_image) {
                html += `<b>Annotated Image:</b><br><img src="data:image/jpeg;base64,${doc.annotated_image}" alt="Annotated" style="max-width:300px;" />`;
            }
            html += `<div style='margin-bottom:10px;color:#666;font-size:0.95em'>Processing method: ${doc.processing_method || 'N/A'}</div>`;
            if (doc.error) {
                html += `<span style='color:red'>Error: ${doc.error}</span>`;
            }
            html += '</li>';
        }
        html += '</ul>';
        batchResultDiv.innerHTML = html;
        
        // Clear selected files after successful submission
        selectedFiles = [];
        updateFileList();
        updateSubmitButton();
    } catch (err) {
        batchResultDiv.innerHTML = `<span style='color:red'>Error: ${err.message}</span>`;
    }
});

let selectedFiles = [];

// Add file button functionality
document.getElementById('addFileBtn').addEventListener('click', function() {
    document.getElementById('singleFile').click();
});

// Handle file selection
document.getElementById('singleFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        // Check if file already exists
        const exists = selectedFiles.some(f => f.name === file.name && f.size === file.size);
        if (!exists) {
            selectedFiles.push(file);
            updateFileList();
            updateSubmitButton();
        } else {
            alert('This file has already been added.');
        }
        // Reset the input
        e.target.value = '';
    }
});

// Update file list display
function updateFileList() {
    const fileListDiv = document.getElementById('fileList');
    if (selectedFiles.length === 0) {
        fileListDiv.innerHTML = '';
        return;
    }

    let html = '<h3>Selected Documents:</h3>';
    selectedFiles.forEach((file, index) => {
        html += `
            <div class="file-item">
                <span class="file-name">📄 ${file.name}</span>
                <span class="file-size">(${(file.size / 1024).toFixed(1)} KB)</span>
                <button type="button" class="remove-file-btn" onclick="removeFile(${index})">❌</button>
            </div>
        `;
    });
    fileListDiv.innerHTML = html;
}

// Remove file from list
function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFileList();
    updateSubmitButton();
}

// Update submit button state
function updateSubmitButton() {
    const submitBtn = document.getElementById('submitBtn');
    const applicantName = document.getElementById('applicantName').value.trim();
    
    if (selectedFiles.length > 0 && applicantName) {
        submitBtn.disabled = false;
        submitBtn.textContent = `📤 Upload & Verify ${selectedFiles.length} Document(s)`;
    } else {
        submitBtn.disabled = true;
        submitBtn.textContent = '📤 Upload & Verify Documents';
    }
}

// Update submit button when applicant name changes
document.getElementById('applicantName').addEventListener('input', updateSubmitButton);
