document.getElementById('batchForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const applicantName = document.getElementById('applicantName').value.trim();
    const batchFiles = document.getElementById('batchFiles');
    const batchResultDiv = document.getElementById('batchResult');
    batchResultDiv.innerHTML = 'Processing...';

    if (!applicantName) {
        batchResultDiv.innerHTML = '<span style="color:red">Please enter applicant name.</span>';
        return;
    }
    if (!batchFiles.files.length) {
        batchResultDiv.innerHTML = '<span style="color:red">Please select at least one file.</span>';
        return;
    }

    const formData = new FormData();
    formData.append('applicant_name', applicantName);
    for (let i = 0; i < batchFiles.files.length; i++) {
        formData.append('files', batchFiles.files[i]);
    }

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
    } catch (err) {
        batchResultDiv.innerHTML = `<span style='color:red'>Error: ${err.message}</span>`;
    }
});
