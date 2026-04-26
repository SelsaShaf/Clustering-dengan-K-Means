// ===== TAB SWITCHING =====
function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
  document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
  document.getElementById(`panel-${tabName}`).classList.add('active');
  document.getElementById('source-input').value = tabName;
}

// ===== K SELECTOR =====
function selectK(k) {
  document.querySelectorAll('.k-opt').forEach(btn => btn.classList.remove('selected'));
  document.querySelector(`[data-k="${k}"]`).classList.add('selected');
  document.getElementById('k-value').value = k;
}

// ===== LOADING =====
function showLoading() {
  document.getElementById('loading-overlay').classList.add('show');
}

// ===== MANUAL TABLE =====
let rowCount = 3;

function addRow() {
  rowCount++;
  const tbody = document.getElementById('manual-tbody');
  const row = document.createElement('tr');
  row.innerHTML = `
    <td><input type="text" name="item_name" placeholder="Nama produk" /></td>
    <td><input type="number" name="calories" placeholder="0" min="0" step="1" /></td>
    <td><input type="number" name="fat" placeholder="0" min="0" step="0.1" /></td>
    <td><input type="number" name="sodium" placeholder="0" min="0" step="1" /></td>
    <td><input type="number" name="carbs" placeholder="0" min="0" step="0.1" /></td>
    <td><input type="number" name="protein" placeholder="0" min="0" step="0.1" /></td>
    <td><button type="button" class="btn-remove-row" onclick="removeRow(this)">✕</button></td>
  `;
  tbody.appendChild(row);
}

function removeRow(btn) {
  const rows = document.querySelectorAll('#manual-tbody tr');
  if (rows.length <= 1) {
    alert('Minimal harus ada 1 baris data!');
    return;
  }
  btn.closest('tr').remove();
}

// ===== UPLOAD FILE NAME =====
document.addEventListener('DOMContentLoaded', function () {
  const fileInput = document.getElementById('csv-file-input');
  if (fileInput) {
    fileInput.addEventListener('change', function () {
      const name = this.files[0] ? this.files[0].name : '';
      const display = document.getElementById('upload-filename');
      if (display) {
        display.textContent = name ? `📄 ${name}` : '';
      }
    });
  }

  // Drag over upload area
  const uploadArea = document.querySelector('.upload-area');
  if (uploadArea) {
    uploadArea.addEventListener('dragover', e => {
      e.preventDefault();
      uploadArea.classList.add('drag-over');
    });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
    uploadArea.addEventListener('drop', e => {
      e.preventDefault();
      uploadArea.classList.remove('drag-over');
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        const display = document.getElementById('upload-filename');
        if (display) display.textContent = `📄 ${e.dataTransfer.files[0].name}`;
      }
    });
  }

  // Form submit → loading
  const form = document.getElementById('cluster-form');
  if (form) {
    form.addEventListener('submit', function (e) {
      // Validasi tab aktif
      const source = document.getElementById('source-input').value;
      if (source === 'manual') {
        const items = document.querySelectorAll('input[name="item_name"]');
        const filled = Array.from(items).filter(i => i.value.trim() !== '');
        if (filled.length === 0) {
          e.preventDefault();
          alert('Masukkan minimal 1 data pada tabel manual!');
          return;
        }
      }
      if (source === 'upload') {
        const f = document.getElementById('csv-file-input');
        if (!f || !f.files.length) {
          e.preventDefault();
          alert('Pilih file CSV terlebih dahulu!');
          return;
        }
      }
      showLoading();
    });
  }
});

// ===== TABLE FILTER & SEARCH (result page) =====
function filterTable() {
  const search = document.getElementById('search-input')?.value.toLowerCase() || '';
  const filterCluster = document.getElementById('filter-cluster')?.value || 'all';
  const rows = document.querySelectorAll('#result-table tbody tr');

  rows.forEach(row => {
    const text = row.textContent.toLowerCase();
    const cluster = row.getAttribute('data-cluster');
    const matchSearch = text.includes(search);
    const matchCluster = filterCluster === 'all' || cluster === filterCluster;
    row.style.display = matchSearch && matchCluster ? '' : 'none';
  });
}
