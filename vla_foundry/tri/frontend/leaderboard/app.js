// Leaderboard Application

const API_BASE = '/api';

// State
let leaderboardData = [];
let selectedRows = new Set();  // Now stores "runId_campaignName" composite keys
let currentSort = { column: 'success_rate', descending: true };
let comparisonChart = null;
let spiderChart = null;
let configSchema = {};
let configFilters = [];
let selectedAblations = new Set();
let allAblations = [];
let masterAblations = [];  // Full list from database
let selectedTasks = new Set();
let allTasks = [];
let masterTasks = [];  // Full list from database
let selectedCampaigns = new Set();
let allCampaigns = [];
let masterCampaigns = [];  // Full list from database
let selectedReference = null;  // Single reference ablation
let allReferences = [];

// Helper to create composite key for row selection
function getRowKey(runId, campaignName) {
    return `${runId}_${campaignName || 'default'}`;
}

function parseRowKey(key) {
    const idx = key.indexOf('_');
    return {
        runId: parseInt(key.substring(0, idx)),
        campaignName: key.substring(idx + 1),
    };
}

// DOM Elements
const elements = {
    campaignChips: document.getElementById('campaign-chips'),
    campaignSearch: document.getElementById('campaign-search'),
    taskChips: document.getElementById('task-chips'),
    taskSearch: document.getElementById('task-search'),
    ablationChips: document.getElementById('ablation-chips'),
    ablationSearch: document.getElementById('ablation-search'),
    referenceSelect: document.getElementById('reference-select'),
    referenceDatalist: document.getElementById('reference-datalist'),
    minRollouts: document.getElementById('min-rollouts'),
    refreshBtn: document.getElementById('refresh-btn'),
    compareBtn: document.getElementById('compare-btn'),
    clearSelectionBtn: document.getElementById('clear-selection-btn'),
    exportBtn: document.getElementById('export-btn'),
    selectAll: document.getElementById('select-all'),
    leaderboardBody: document.getElementById('leaderboard-body'),
    selectedCount: document.getElementById('selected-count'),
    compareModal: document.getElementById('compare-modal'),
    closeCompare: document.getElementById('close-compare'),
    configModal: document.getElementById('config-modal'),
    closeConfig: document.getElementById('close-config'),
    configContent: document.getElementById('config-content'),
    totalRuns: document.getElementById('total-runs'),
    totalRollouts: document.getElementById('total-rollouts'),
    avgSuccess: document.getElementById('avg-success'),
    totalTasks: document.getElementById('total-tasks'),
    addConfigFilterBtn: document.getElementById('add-config-filter-btn'),
    configFilterRows: document.getElementById('config-filter-rows'),
    configDiffContainer: document.getElementById('config-diff-container'),
};

// JSON Syntax Highlighting
function highlightJSON(json) {
    if (typeof json !== 'string') {
        json = JSON.stringify(json, null, 2);
    }

    // Escape HTML
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    // Apply syntax highlighting
    return json.replace(
        /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        function (match) {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                    // Remove the colon for keys, we'll add it back
                    match = match.slice(0, -1);
                    return '<span class="' + cls + '">' + match + '</span>:';
                } else {
                    cls = 'json-string';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-boolean';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
            return '<span class="' + cls + '">' + match + '</span>';
        }
    );
}

// API Functions
async function fetchJSON(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
}

async function fetchLeaderboard() {
    const params = new URLSearchParams();

    // Handle selected campaigns from chips
    if (selectedCampaigns.size > 0) {
        selectedCampaigns.forEach(campaign => {
            params.append('campaign', campaign);
        });
    }
    // Handle selected tasks from chips
    if (selectedTasks.size > 0) {
        selectedTasks.forEach(task => {
            params.append('task', task);
        });
    }
    // Handle selected ablations from chips
    // Also include reference ablation if set (needed for comparison)
    const ablationsToFetch = new Set(selectedAblations);
    if (selectedReference) {
        ablationsToFetch.add(selectedReference);
    }
    if (ablationsToFetch.size > 0) {
        ablationsToFetch.forEach(ablation => {
            params.append('ablation', ablation);
        });
    }
    if (elements.minRollouts.value) {
        params.set('min_rollouts', elements.minRollouts.value);
    }
    // Add config filters
    if (configFilters.length > 0) {
        params.set('config_filters', JSON.stringify(configFilters));
    }
    params.set('order_by', currentSort.column);
    params.set('descending', currentSort.descending);

    const result = await fetchJSON(`${API_BASE}/leaderboard?${params}`);
    return result.data;
}

async function fetchConfigSchema() {
    const result = await fetchJSON(`${API_BASE}/config-schema`);
    return result.data;
}

async function fetchConfigDiff(runIds) {
    const response = await fetch(`${API_BASE}/config-diff`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_ids: runIds }),
    });
    const result = await response.json();
    return result.data;
}

async function fetchFilters() {
    const [campaigns, tasks, ablations] = await Promise.all([
        fetchJSON(`${API_BASE}/campaigns`),
        fetchJSON(`${API_BASE}/tasks`),
        fetchJSON(`${API_BASE}/ablations`),
    ]);
    return {
        campaigns: campaigns.data,
        tasks: tasks.data,
        ablations: ablations.data,
    };
}

async function fetchStats() {
    const result = await fetchJSON(`${API_BASE}/stats`);
    return result.data;
}

async function fetchConfig(runId) {
    const result = await fetchJSON(`${API_BASE}/config/${runId}`);
    return result.data;
}

async function fetchComparison(selections) {
    // selections is an array of {runId, campaignName} objects
    const response = await fetch(`${API_BASE}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selections }),
    });
    const result = await response.json();
    return result.data;
}

// UI Functions
function populateSelect(select, items, valueKey = null) {
    // For multi-select, preserve current selections
    const isMultiple = select.multiple;
    const currentSelections = isMultiple
        ? [...select.selectedOptions].map(o => o.value)
        : [select.value];

    select.innerHTML = '';

    // Only add "All" option for single-select
    if (!isMultiple) {
        const firstOption = document.createElement('option');
        firstOption.value = '';
        firstOption.textContent = `All ${select.id.replace('-filter', '').replace(/^\w/, c => c.toUpperCase())}s`;
        select.appendChild(firstOption);
    }

    items.forEach(item => {
        const option = document.createElement('option');
        option.value = valueKey ? item[valueKey] : item;
        option.textContent = valueKey ? item[valueKey] : item;
        select.appendChild(option);
    });

    // Restore selections
    if (isMultiple) {
        [...select.options].forEach(o => {
            if (currentSelections.includes(o.value)) {
                o.selected = true;
            }
        });
    } else if ([...select.options].some(o => o.value === currentSelections[0])) {
        select.value = currentSelections[0];
    }
}

// Reference Ablation Select Functions (single selection with searchable datalist)
function populateReferenceDatalist(ablations, availableAblations = null) {
    allReferences = ablations;
    elements.referenceDatalist.innerHTML = '';

    // If availableAblations is provided, only show those options
    const ablationsToShow = availableAblations ? ablations.filter(a => availableAblations.has(a)) : ablations;

    ablationsToShow.forEach(ablation => {
        const option = document.createElement('option');
        option.value = ablation;
        elements.referenceDatalist.appendChild(option);
    });

    // Clear selection if current reference is not in available list
    if (selectedReference && availableAblations && !availableAblations.has(selectedReference)) {
        selectedReference = null;
        elements.referenceSelect.value = '';
    }
}

// Ablation Chip Functions
function renderAblationChips(ablations, availableAblations = null) {
    allAblations = ablations;
    elements.ablationChips.innerHTML = '';

    // If availableAblations is provided, only show those chips
    const ablationsToShow = availableAblations ? ablations.filter(a => availableAblations.has(a)) : ablations;

    ablationsToShow.forEach(ablation => {
        const chip = document.createElement('span');
        chip.className = 'chip' + (selectedAblations.has(ablation) ? ' selected' : '');
        chip.textContent = ablation;
        chip.dataset.ablation = ablation;

        chip.addEventListener('click', () => {
            if (selectedAblations.has(ablation)) {
                selectedAblations.delete(ablation);
                chip.classList.remove('selected');
            } else {
                selectedAblations.add(ablation);
                chip.classList.add('selected');
            }
            refresh();
        });

        elements.ablationChips.appendChild(chip);
    });
}

function selectAllAblations() {
    // Only select visible chips
    document.querySelectorAll('#ablation-chips .chip').forEach(chip => {
        selectedAblations.add(chip.dataset.ablation);
        chip.classList.add('selected');
    });
    refresh();
}

function clearAllAblations() {
    selectedAblations.clear();
    document.querySelectorAll('#ablation-chips .chip').forEach(chip => {
        chip.classList.remove('selected');
    });
    refresh();
}

function filterAblationChips(searchTerm) {
    const term = searchTerm.toLowerCase();
    document.querySelectorAll('#ablation-chips .chip').forEach(chip => {
        const ablation = chip.dataset.ablation.toLowerCase();
        if (term === '' || ablation.includes(term)) {
            chip.style.display = '';
        } else {
            chip.style.display = 'none';
        }
    });
}

// Task Chip Functions
function renderTaskChips(tasks, availableTasks = null) {
    allTasks = tasks;
    elements.taskChips.innerHTML = '';

    // If availableTasks is provided, only show those chips
    const tasksToShow = availableTasks ? tasks.filter(t => availableTasks.has(t)) : tasks;

    tasksToShow.forEach(task => {
        const chip = document.createElement('span');
        chip.className = 'chip' + (selectedTasks.has(task) ? ' selected' : '');
        chip.textContent = task;
        chip.dataset.task = task;

        chip.addEventListener('click', () => {
            if (selectedTasks.has(task)) {
                selectedTasks.delete(task);
                chip.classList.remove('selected');
            } else {
                selectedTasks.add(task);
                chip.classList.add('selected');
            }
            refresh();
        });

        elements.taskChips.appendChild(chip);
    });
}

function selectAllTasks() {
    // Only select visible chips
    document.querySelectorAll('#task-chips .chip').forEach(chip => {
        selectedTasks.add(chip.dataset.task);
        chip.classList.add('selected');
    });
    refresh();
}

function clearAllTasks() {
    selectedTasks.clear();
    document.querySelectorAll('#task-chips .chip').forEach(chip => {
        chip.classList.remove('selected');
    });
    refresh();
}

function filterTaskChips(searchTerm) {
    const term = searchTerm.toLowerCase();
    document.querySelectorAll('#task-chips .chip').forEach(chip => {
        const task = chip.dataset.task.toLowerCase();
        if (term === '' || task.includes(term)) {
            chip.style.display = '';
        } else {
            chip.style.display = 'none';
        }
    });
}

// Campaign Chip Functions
function renderCampaignChips(campaigns, availableCampaigns = null) {
    allCampaigns = campaigns;
    elements.campaignChips.innerHTML = '';

    // If availableCampaigns is provided, only show those chips
    const campaignsToShow = availableCampaigns ? campaigns.filter(c => availableCampaigns.has(c)) : campaigns;

    campaignsToShow.forEach(campaign => {
        const chip = document.createElement('span');
        chip.className = 'chip' + (selectedCampaigns.has(campaign) ? ' selected' : '');
        chip.textContent = campaign;
        chip.dataset.campaign = campaign;

        chip.addEventListener('click', () => {
            if (selectedCampaigns.has(campaign)) {
                selectedCampaigns.delete(campaign);
                chip.classList.remove('selected');
            } else {
                selectedCampaigns.add(campaign);
                chip.classList.add('selected');
            }
            refresh();
        });

        elements.campaignChips.appendChild(chip);
    });
}

function selectAllCampaigns() {
    // Only select visible chips
    document.querySelectorAll('#campaign-chips .chip').forEach(chip => {
        selectedCampaigns.add(chip.dataset.campaign);
        chip.classList.add('selected');
    });
    refresh();
}

function clearAllCampaigns() {
    selectedCampaigns.clear();
    document.querySelectorAll('#campaign-chips .chip').forEach(chip => {
        chip.classList.remove('selected');
    });
    refresh();
}

function filterCampaignChips(searchTerm) {
    const term = searchTerm.toLowerCase();
    document.querySelectorAll('#campaign-chips .chip').forEach(chip => {
        const campaign = chip.dataset.campaign.toLowerCase();
        if (term === '' || campaign.includes(term)) {
            chip.style.display = '';
        } else {
            chip.style.display = 'none';
        }
    });
}

// Config Filter Functions (Query Builder)
function getFieldsByCategory() {
    const fieldPaths = Object.keys(configSchema).sort();
    const categories = {};

    fieldPaths.forEach(path => {
        const parts = path.split('.');
        const category = parts[0];
        const displayName = parts.slice(1).join('.') || path;

        if (!categories[category]) {
            categories[category] = [];
        }
        categories[category].push({ path, displayName });
    });

    return categories;
}

function createConfigFilterRow() {
    const row = document.createElement('div');
    row.className = 'config-filter-row';

    const categories = getFieldsByCategory();
    const sortedCategories = Object.keys(categories).sort();

    let optionsHtml = '<option value="">Select field...</option>';
    sortedCategories.forEach(category => {
        optionsHtml += `<optgroup label="${category}">`;
        categories[category].forEach(({ path, displayName }) => {
            optionsHtml += `<option value="${path}">${displayName}</option>`;
        });
        optionsHtml += '</optgroup>';
    });

    row.innerHTML = `
        <select class="field-select">
            ${optionsHtml}
        </select>
        <select class="op-select">
            <option value="=">=</option>
            <option value="!=">!=</option>
            <option value=">">&gt;</option>
            <option value="<">&lt;</option>
            <option value=">=">&gt;=</option>
            <option value="<=">&lt;=</option>
            <option value="contains">contains</option>
        </select>
        <input type="text" class="value-input" placeholder="Value...">
        <button class="remove-filter-btn">&times;</button>
    `;

    const fieldSelect = row.querySelector('.field-select');
    const valueInput = row.querySelector('.value-input');

    // When field changes, show available values as datalist
    fieldSelect.addEventListener('change', () => {
        const field = fieldSelect.value;
        if (field && configSchema[field]) {
            const values = configSchema[field].values;
            // Create or update datalist
            let datalistId = `values-${field.replace(/\./g, '-')}`;
            let datalist = document.getElementById(datalistId);
            if (!datalist) {
                datalist = document.createElement('datalist');
                datalist.id = datalistId;
                document.body.appendChild(datalist);
            }
            datalist.innerHTML = values.map(v =>
                `<option value="${v.value}">${v.value} (${v.count} runs)</option>`
            ).join('');
            valueInput.setAttribute('list', datalistId);
        }
        updateConfigFilters();
    });

    valueInput.addEventListener('change', updateConfigFilters);
    row.querySelector('.op-select').addEventListener('change', updateConfigFilters);

    row.querySelector('.remove-filter-btn').addEventListener('click', () => {
        row.remove();
        updateConfigFilters();
    });

    return row;
}

function updateConfigFilters() {
    configFilters = [];
    const rows = elements.configFilterRows.querySelectorAll('.config-filter-row');

    rows.forEach(row => {
        const field = row.querySelector('.field-select').value;
        const op = row.querySelector('.op-select').value;
        const value = row.querySelector('.value-input').value;

        if (field && value) {
            configFilters.push({ field, op, value });
        }
    });

    refresh();
}

function addConfigFilterRow() {
    const row = createConfigFilterRow();
    elements.configFilterRows.appendChild(row);
}

function getSuccessClass(rate) {
    if (rate >= 0.7) return 'high';
    if (rate >= 0.4) return 'medium';
    return 'low';
}

function formatDate(dateString) {
    if (!dateString) return '-';
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
        });
    } catch {
        return '-';
    }
}

function formatS3Url(s3Path) {
    // Convert s3://bucket/path to custom S3 browser URL
    if (!s3Path) return '#';
    return `http://10.161.51.141:8080/${s3Path}`;
}

// Build reference lookup: task_name -> success_rate for reference ablation
function buildReferenceLookup(data, refAblation) {
    const lookup = {};
    if (!refAblation) return lookup;

    data.forEach(row => {
        if (row.ablation === refAblation) {
            lookup[row.task_name] = row.success_rate;
        }
    });
    return lookup;
}

function getS3RunRoot(s3Path) {
    if (!s3Path) return s3Path;
    let p = String(s3Path).replace(/\/+$/, '');
    const checkpointsIdx = p.indexOf('/checkpoints/');
    if (checkpointsIdx !== -1) {
        return p.slice(0, checkpointsIdx) + '/';
    }
    if (p.endsWith('/checkpoints')) {
        return p.slice(0, -'/checkpoints'.length) + '/';
    }
    return p + '/';
}

function renderLeaderboard(fullData, displayData) {
    leaderboardData = fullData;

    if (displayData.length === 0) {
        elements.leaderboardBody.innerHTML = `
            <tr><td colspan="8" class="loading">No results found</td></tr>
        `;
        return;
    }

    // Build reference lookup from full data (includes reference ablation)
    const refLookup = buildReferenceLookup(fullData, selectedReference);
    const hasReference = selectedReference !== null;

    elements.leaderboardBody.innerHTML = displayData.map(row => {
        const successClass = getSuccessClass(row.success_rate);
        const rowKey = getRowKey(row.run_id, row.campaign_name);
        const isSelected = selectedRows.has(rowKey);

        // Build success rate cell content
        let successCellHtml = '';
        if (hasReference) {
            // Reference mode: show diff as main value with comparison bar
            if (row.ablation === selectedReference) {
                // This is the reference row
                successCellHtml = `
                    <span class="diff-value reference">ref</span>
                    <span class="absolute-rate">(${(row.success_rate * 100).toFixed(1)}%)</span>
                    <div class="progress-bar">
                        <div class="progress-fill ${successClass}"
                             style="width: ${row.success_rate * 100}%"></div>
                    </div>
                `;
            } else if (refLookup[row.task_name] !== undefined) {
                const diff = (row.success_rate - refLookup[row.task_name]) * 100;
                const diffClass = diff > 0 ? 'positive' : diff < 0 ? 'negative' : 'reference';
                const diffSign = diff > 0 ? '+' : '';
                // Scale bar: 50pp = full half bar, clamp to reasonable range
                const barWidth = Math.min(Math.abs(diff) * 2, 50); // max 50% of bar width
                successCellHtml = `
                    <span class="diff-value ${diffClass}">${diffSign}${diff.toFixed(1)}pp</span>
                    <span class="absolute-rate">(${(row.success_rate * 100).toFixed(1)}%)</span>
                    <div class="comparison-bar">
                        <div class="comparison-bar-center"></div>
                        <div class="comparison-bar-fill ${diffClass}" style="width: ${barWidth}%"></div>
                    </div>
                `;
            } else {
                // No reference data for this task
                successCellHtml = `
                    <span class="diff-value na">n/a</span>
                    <span class="absolute-rate">(${(row.success_rate * 100).toFixed(1)}%)</span>
                    <div class="progress-bar">
                        <div class="progress-fill ${successClass}"
                             style="width: ${row.success_rate * 100}%"></div>
                    </div>
                `;
            }
        } else {
            // Normal mode: show success rate with standard bar
            successCellHtml = `
                <span class="success-rate ${successClass}">
                    ${(row.success_rate * 100).toFixed(1)}%
                </span>
                <div class="progress-bar">
                    <div class="progress-fill ${successClass}"
                         style="width: ${row.success_rate * 100}%"></div>
                </div>
            `;
        }

        return `
            <tr data-row-key="${rowKey}">
                <td class="checkbox-col">
                    <input type="checkbox" class="row-checkbox"
                           data-row-key="${rowKey}"
                           data-run-id="${row.run_id}"
                           ${isSelected ? 'checked' : ''}>
                </td>
                <td>
                    ${successCellHtml}
                </td>
                <td class="task-name" title="${row.task_name}">${row.task_name}</td>
                <td>
                    ${row.ablation ? `<span class="ablation-tag">${row.ablation}</span>` : '-'}
                </td>
                <td class="campaign-name">${row.campaign_name || '-'}</td>
                <td>${row.successes}/${row.total_rollouts}</td>
                <td class="date-added">${formatDate(row.evaluated_at)}</td>
                <td class="links">
                    ${row.has_config
                ? `<a href="#" class="link-btn config-link" data-run-id="${row.run_id}">Config</a>`
                : `<span class="link-btn disabled">Config</span>`
            }
                    ${row.s3_path
                ? `<a href="http://10.161.51.141:8080/${getS3RunRoot(row.s3_path)}" target="_blank" class="link-btn">S3</a>`
                : `<span class="link-btn disabled">S3</span>`
            }
                    ${row.wandb_link && row.has_wandb
                ? `<a href="${row.wandb_link}" target="_blank" class="link-btn">W&B</a>`
                : `<span class="link-btn disabled">W&B</span>`
            }
                </td>
            </tr>
        `;
    }).join('');

    // Add event listeners for checkboxes
    document.querySelectorAll('.row-checkbox').forEach(cb => {
        cb.addEventListener('change', handleRowSelect);
    });

    // Add event listeners for config links
    document.querySelectorAll('.config-link').forEach(link => {
        link.addEventListener('click', handleConfigClick);
    });

    updateSelectionUI();
}

function updateSelectionUI() {
    elements.selectedCount.textContent = selectedRows.size;
    elements.compareBtn.disabled = selectedRows.size < 2;
    elements.selectAll.checked = leaderboardData.length > 0 &&
        leaderboardData.every(row => selectedRows.has(getRowKey(row.run_id, row.campaign_name)));
}

function updateStats(currentData, totalStats) {
    // Calculate stats from current filtered view
    const currentRuns = currentData.length;
    const currentRollouts = currentData.reduce((sum, d) => sum + d.total_rollouts, 0);
    const currentAvgSuccess = currentData.length > 0
        ? currentData.reduce((sum, d) => sum + d.success_rate, 0) / currentData.length
        : 0;
    const currentTasks = new Set(currentData.map(d => d.task_name)).size;

    // Display as current / total
    elements.totalRuns.textContent = `${currentRuns.toLocaleString()} / ${totalStats.total_runs.toLocaleString()}`;
    elements.totalRollouts.textContent = `${currentRollouts.toLocaleString()} / ${totalStats.total_rollouts.toLocaleString()}`;
    elements.avgSuccess.textContent = `${(currentAvgSuccess * 100).toFixed(1)}%`;
    elements.totalTasks.textContent = `${currentTasks} / ${totalStats.tasks}`;
}

function updateSortIndicators() {
    document.querySelectorAll('th.sortable').forEach(th => {
        th.classList.remove('asc', 'desc');
        if (th.dataset.sort === currentSort.column) {
            th.classList.add(currentSort.descending ? 'desc' : 'asc');
        }
    });
}

// Event Handlers
function handleRowSelect(e) {
    const rowKey = e.target.dataset.rowKey;
    if (e.target.checked) {
        selectedRows.add(rowKey);
    } else {
        selectedRows.delete(rowKey);
    }
    updateSelectionUI();
    updateUrlParams();
}

function handleSelectAll(e) {
    if (e.target.checked) {
        leaderboardData.forEach(row => {
            const rowKey = getRowKey(row.run_id, row.campaign_name);
            selectedRows.add(rowKey);
        });
    } else {
        leaderboardData.forEach(row => {
            const rowKey = getRowKey(row.run_id, row.campaign_name);
            selectedRows.delete(rowKey);
        });
    }
    document.querySelectorAll('.row-checkbox').forEach(cb => {
        cb.checked = e.target.checked;
    });
    updateSelectionUI();
    updateUrlParams();
}

async function handleConfigClick(e) {
    e.preventDefault();
    const runId = parseInt(e.target.dataset.runId);

    try {
        const config = await fetchConfig(runId);
        // Use syntax-highlighted JSON
        elements.configContent.innerHTML = highlightJSON(config);
        elements.configModal.classList.remove('hidden');
    } catch (error) {
        alert('Failed to load config: ' + error.message);
    }
}

function handleSort(e) {
    const column = e.target.dataset.sort;
    if (!column) return;

    if (currentSort.column === column) {
        currentSort.descending = !currentSort.descending;
    } else {
        currentSort.column = column;
        currentSort.descending = true;
    }

    updateSortIndicators();
    refresh();
}

async function handleCompare() {
    // Extract selections as {runId, campaignName} objects
    const selections = Array.from(selectedRows).map(key => parseRowKey(key));
    // Also get unique run_ids for config diff
    const runIds = [...new Set(selections.map(s => s.runId))];

    // Show modal immediately with loading state
    elements.compareModal.classList.remove('hidden');
    document.getElementById('violin-chart').innerHTML = '<div class="loading-spinner">Computing statistics...</div>';
    document.getElementById('aggregate-violin-chart').innerHTML = '';
    document.getElementById('scatter-chart').innerHTML = '<div class="loading-spinner">Loading...</div>';
    elements.configDiffContainer.innerHTML = '<p class="loading">Loading config differences...</p>';

    try {
        // Fetch comparison data, config diff, and stats in parallel
        const [comparisonData, diffData, newStatsData] = await Promise.all([
            fetchComparison(selections),
            fetchConfigDiff(runIds),
            fetchCompareStats(selections),
        ]);

        // Cache stats data for scatter plot updates
        statsData = newStatsData;

        showComparisonModal(comparisonData);
        renderConfigDiff(diffData);

        // Render statistical plots (or show error message)
        if (statsData && statsData.error) {
            // Show error message in violin and scatter tabs
            document.getElementById('violin-chart').innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #666;">
                    <div style="text-align: center; padding: 20px;">
                        <p style="font-size: 1.2em; margin-bottom: 10px;">⚠️ ${statsData.message}</p>
                        <p style="font-size: 0.9em;">Try selecting fewer runs or filtering by ablation.</p>
                    </div>
                </div>
            `;
            document.getElementById('aggregate-violin-chart').innerHTML = '';
            document.getElementById('scatter-chart').innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #666;">
                    <div style="text-align: center; padding: 20px;">
                        <p style="font-size: 1.2em;">⚠️ Statistical comparison unavailable</p>
                    </div>
                </div>
            `;
        } else if (statsData) {
            renderViolinPlot(statsData);
            setupScatterControls(statsData);
        }

        // Reset to chart tab
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelector('.tab-btn[data-tab="chart"]').classList.add('active');
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById('tab-chart').classList.add('active');
    } catch (error) {
        alert('Failed to compare runs: ' + error.message);
    }
}

function showComparisonModal(data) {
    elements.compareModal.classList.remove('hidden');

    // Group data by task
    const taskGroups = {};
    data.forEach(row => {
        if (!taskGroups[row.task_name]) {
            taskGroups[row.task_name] = [];
        }
        taskGroups[row.task_name].push(row);
    });

    // Create bar chart
    const ctx = document.getElementById('comparison-chart').getContext('2d');

    if (comparisonChart) {
        comparisonChart.destroy();
    }

    const labels = Object.keys(taskGroups);
    const datasets = [];

    // Get unique combinations of (ablation, campaign) for cross-campaign comparison
    const campaigns = [...new Set(data.map(d => d.campaign_name || 'default'))];
    const ablations = [...new Set(data.map(d => d.ablation || 'default'))];

    // Create series for each ablation+campaign combination
    const seriesKeys = [];
    data.forEach(d => {
        const key = campaigns.length > 1
            ? `${d.ablation || 'default'} (${d.campaign_name || 'default'})`
            : (d.ablation || 'default');
        if (!seriesKeys.includes(key)) {
            seriesKeys.push(key);
        }
    });

    const colors = [
        '#2563eb', '#dc2626', '#16a34a', '#ca8a04',
        '#9333ea', '#0891b2', '#be185d', '#65a30d',
        '#4f46e5', '#e11d48', '#059669', '#d97706'
    ];

    // Build reference lookup for comparison modal (task -> success_rate)
    const refLookup = {};
    const hasReference = selectedReference !== null;
    if (hasReference) {
        data.forEach(row => {
            if (row.ablation === selectedReference) {
                refLookup[row.task_name] = row.success_rate;
            }
        });
    }

    // Determine if we're in relative mode (reference selected and has data)
    const useRelativeMode = hasReference && Object.keys(refLookup).length > 0;

    // Add "Mean" to labels if multiple tasks
    const showMean = labels.length > 1;
    const chartLabels = showMean ? [...labels, 'Mean'] : labels;

    seriesKeys.forEach((seriesKey, i) => {
        const values = labels.map(task => {
            const entry = taskGroups[task].find(d => {
                const rowKey = campaigns.length > 1
                    ? `${d.ablation || 'default'} (${d.campaign_name || 'default'})`
                    : (d.ablation || 'default');
                return rowKey === seriesKey;
            });
            if (!entry) return null;

            if (useRelativeMode) {
                // Relative mode: show difference from reference
                const refRate = refLookup[task];
                if (refRate === undefined) return null;
                // Return difference in percentage points
                return (entry.success_rate - refRate) * 100;
            } else {
                // Absolute mode
                return entry.success_rate * 100;
            }
        });

        // Calculate mean if showing mean bar
        if (showMean) {
            const validValues = values.filter(v => v !== null);
            const mean = validValues.length > 0
                ? validValues.reduce((sum, v) => sum + v, 0) / validValues.length
                : null;
            values.push(mean);
        }

        datasets.push({
            label: seriesKey,
            data: values,
            backgroundColor: colors[i % colors.length],
            borderColor: colors[i % colors.length],
            borderWidth: 1,
        });
    });

    // Chart options differ based on mode
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            title: {
                display: true,
                text: useRelativeMode
                    ? `Success Rate Difference vs Reference (${selectedReference})`
                    : 'Success Rate Comparison by Task',
            },
            legend: {
                position: 'top',
            },
            tooltip: useRelativeMode ? {
                callbacks: {
                    label: function (context) {
                        const value = context.parsed.y;
                        if (value === null) return '';
                        const sign = value >= 0 ? '+' : '';
                        return `${context.dataset.label}: ${sign}${value.toFixed(1)}pp`;
                    }
                }
            } : {},
        },
        scales: {
            y: useRelativeMode ? {
                // Relative mode: center at zero, symmetric range
                title: {
                    display: true,
                    text: 'Difference (percentage points)',
                },
            } : {
                beginAtZero: true,
                max: 100,
                title: {
                    display: true,
                    text: 'Success Rate (%)',
                },
            },
        },
    };

    // Add minBarLength to make zero-value bars visible and hoverable
    chartOptions.datasets = {
        bar: {
            minBarLength: 4,  // Minimum 4px height for zero-value bars
        }
    };

    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: chartLabels, datasets },
        options: chartOptions,
    });

    // Create spider/radar chart
    const spiderCtx = document.getElementById('spider-chart').getContext('2d');

    if (spiderChart) {
        spiderChart.destroy();
    }

    // For spider chart, always show absolute values but highlight reference
    const spiderDatasets = seriesKeys.map((seriesKey, i) => {
        const values = labels.map(task => {
            const entry = taskGroups[task].find(d => {
                const rowKey = campaigns.length > 1
                    ? `${d.ablation || 'default'} (${d.campaign_name || 'default'})`
                    : (d.ablation || 'default');
                return rowKey === seriesKey;
            });
            return entry ? entry.success_rate * 100 : 0;
        });

        // Check if this series is the reference ablation
        const isReference = hasReference && seriesKey === selectedReference;

        return {
            label: isReference ? `${seriesKey} (ref)` : seriesKey,
            data: values,
            backgroundColor: colors[i % colors.length] + (isReference ? '15' : '33'), // Less opaque for reference (renders at back)
            borderColor: colors[i % colors.length],
            borderWidth: isReference ? 3 : 2, // Thicker border for reference
            borderDash: isReference ? [5, 5] : [], // Dashed line for reference
            pointBackgroundColor: colors[i % colors.length],
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: colors[i % colors.length],
            pointRadius: isReference ? 5 : 3, // Larger points for reference
        };
    });

    // Move reference dataset to back so other runs render on top
    if (hasReference) {
        const refIdx = spiderDatasets.findIndex(d => d.label.includes('(ref)'));
        if (refIdx > 0) {
            const [refDataset] = spiderDatasets.splice(refIdx, 1);
            spiderDatasets.unshift(refDataset);
        }
    }

    spiderChart = new Chart(spiderCtx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: spiderDatasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: hasReference
                        ? `Success Rate by Task (Reference: ${selectedReference})`
                        : 'Success Rate by Task (Spider Chart)',
                },
                legend: {
                    position: 'top',
                },
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                    },
                    pointLabels: {
                        font: {
                            size: 11,
                        },
                    },
                },
            },
        },
    });

    // Create comparison table
    const tableHtml = `
        <table>
            <thead>
                <tr>
                    <th>Task</th>
                    <th>Ablation</th>
                    <th>Campaign</th>
                    <th>Success Rate</th>
                    <th>Rollouts</th>
                </tr>
            </thead>
            <tbody>
                ${data.map(row => `
                    <tr>
                        <td>${row.task_name}</td>
                        <td>${row.ablation || 'default'}</td>
                        <td>${row.campaign_name || 'default'}</td>
                        <td class="success-rate ${getSuccessClass(row.success_rate)}">
                            ${(row.success_rate * 100).toFixed(1)}%
                        </td>
                        <td>${row.successes}/${row.total_rollouts}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    document.getElementById('comparison-table-container').innerHTML = tableHtml;
}

async function handleExport() {
    try {
        const data = await fetchJSON(`${API_BASE}/export`);
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'leaderboard_export.json';
        a.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        alert('Failed to export: ' + error.message);
    }
}

// Config Diff Display
function renderConfigDiff(diffData) {
    if (!diffData.fields || diffData.fields.length === 0) {
        elements.configDiffContainer.innerHTML = '<p class="loading">No config differences found between selected runs.</p>';
        return;
    }

    const runs = diffData.runs;
    const fields = diffData.fields;

    // Sort runs by success rate (highest first)
    runs.sort((a, b) => (b.success_rate || 0) - (a.success_rate || 0));

    // Group fields by top-level category
    const groupedFields = {};
    fields.forEach(field => {
        const parts = field.field_path.split('.');
        const category = parts[0];
        const subPath = parts.slice(1).join('.') || parts[0];

        if (!groupedFields[category]) {
            groupedFields[category] = [];
        }
        groupedFields[category].push({
            ...field,
            subPath: subPath,
        });
    });

    const categories = Object.keys(groupedFields).sort();
    const numCols = runs.length + 1;

    // Color palette for unique values
    const valueColors = [
        '#e3f2fd', '#fff3e0', '#e8f5e9', '#fce4ec', '#f3e5f5',
        '#e0f7fa', '#fff8e1', '#f1f8e9', '#ffebee', '#ede7f6'
    ];

    // Helper to get color for a value within a row
    // Returns null if there are more unique values than colors available
    function getValueColorMap(values, runIds) {
        const uniqueValues = [...new Set(runIds.map(id => values[id] || '-'))];
        // Don't color if too many unique values (would reuse colors confusingly)
        if (uniqueValues.length > valueColors.length) {
            return null;
        }
        const colorMap = {};
        uniqueValues.forEach((val, i) => {
            colorMap[val] = valueColors[i];
        });
        return colorMap;
    }

    let html = `
        <div style="overflow-x: auto;">
        <table class="config-diff-table">
            <thead>
                <tr>
                    <th>Config Field</th>
                    ${runs.map(r => `
                        <th class="run-header">
                            ${r.scenario_name || `Run ${r.run_id}`}
                            <span class="success-badge ${getSuccessClass(r.success_rate || 0)}">
                                ${r.success_rate ? (r.success_rate * 100).toFixed(1) + '%' : 'N/A'}
                            </span>
                        </th>
                    `).join('')}
                </tr>
            </thead>
            <tbody>
    `;

    const runIds = runs.map(r => r.run_id);

    categories.forEach(category => {
        const categoryFields = groupedFields[category];
        const categoryId = `config-group-${category}`;

        // Group header row (collapsible)
        html += `
            <tr class="config-group-header" data-group="${categoryId}" onclick="toggleConfigGroup('${categoryId}')">
                <td colspan="${numCols}">
                    <span class="group-toggle">▼</span>
                    <strong>${category}</strong>
                    <span class="group-count">(${categoryFields.length} fields)</span>
                </td>
            </tr>
        `;

        // Field rows within group
        categoryFields.forEach(field => {
            const colorMap = getValueColorMap(field.values, runIds);
            const uniqueCount = colorMap ? Object.keys(colorMap).length : 0;

            html += `
                <tr class="config-group-row" data-group="${categoryId}">
                    <td class="field-path field-subpath">
                        ${field.subPath}
                        ${uniqueCount === 2 ? '<span class="diff-indicator">●</span>' : ''}
                    </td>
                    ${runs.map(r => {
                const value = field.values[r.run_id] || '-';
                const bgColor = (colorMap && uniqueCount > 1) ? colorMap[value] : '';
                const style = bgColor ? `style="background: ${bgColor}"` : '';
                return `<td class="diff-value" ${style} title="${value}">${value}</td>`;
            }).join('')}
                </tr>
            `;
        });
    });

    html += `</tbody></table></div>`;
    elements.configDiffContainer.innerHTML = html;
}

// Toggle config group visibility
function toggleConfigGroup(groupId) {
    const header = document.querySelector(`.config-group-header[data-group="${groupId}"]`);
    const rows = document.querySelectorAll(`.config-group-row[data-group="${groupId}"]`);
    const toggle = header.querySelector('.group-toggle');

    const isCollapsed = header.classList.toggle('collapsed');
    toggle.textContent = isCollapsed ? '▶' : '▼';

    rows.forEach(row => {
        row.style.display = isCollapsed ? 'none' : '';
    });
}

// ============================================================================
// Statistical Plots (Violin & Scatter)
// ============================================================================

let statsData = null;  // Cache for stats API response

async function fetchCompareStats(selections) {
    const response = await fetch(`${API_BASE}/compare/stats`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selections, n_samples: 500, max_per_task: 5 }),
    });
    const result = await response.json();
    if (!response.ok) {
        // Return error info instead of throwing
        return { error: true, message: result.detail || 'Failed to compute statistics' };
    }
    return result.data;
}

// Color palette matching the paper
const VIOLIN_COLORS = [
    '#4DBBD5',  // Cyan
    '#E64B35',  // Red
    '#00A087',  // Teal
    '#3C5488',  // Blue
    '#F39B7F',  // Salmon
    '#8491B4',  // Gray-blue
    '#91D1C2',  // Light teal
    '#DC0000',  // Bright red
    '#7E6148',  // Brown
    '#B09C85',  // Tan
];

function renderViolinPlot(statsData) {
    const { by_task, task_order, aggregate } = statsData;

    // Get unique ablations
    const ablations = [...new Set(aggregate.map(a => a.ablation))];
    const ablationColorMap = {};
    ablations.forEach((abl, i) => {
        ablationColorMap[abl] = VIOLIN_COLORS[i % VIOLIN_COLORS.length];
    });

    // Combined plot: per-task violins + separator + aggregate
    const traces = [];
    let showLegend = new Set();
    const nAbl = ablations.length;
    const groupWidth = 0.8;
    const violinWidth = Math.min(0.35, groupWidth / nAbl);

    // Per-task violins
    task_order.forEach((task, taskIdx) => {
        const taskResults = by_task[task] || [];

        ablations.forEach((ablation, ablIdx) => {
            const result = taskResults.find(r => r.ablation === ablation);
            if (result && result.violin_samples.length > 0) {
                const samples = result.violin_samples;
                const sorted = [...samples].sort((a, b) => a - b);
                const q5 = sorted[Math.floor(sorted.length * 0.05)];
                const q95 = sorted[Math.floor(sorted.length * 0.95)];
                const successRate = result.success_rate;  // Use raw success rate

                const isFirstForAblation = !showLegend.has(ablation);
                showLegend.add(ablation);

                const offset = (ablIdx - (nAbl - 1) / 2) * (groupWidth / nAbl);
                const xPos = taskIdx + offset;

                const hoverText = `${(successRate * 100).toFixed(1)}% [${(q5 * 100).toFixed(0)}-${(q95 * 100).toFixed(0)}]`;
                traces.push({
                    type: 'violin',
                    y: samples,
                    x: samples.map(() => xPos),
                    name: ablation,
                    legendgroup: ablation,
                    showlegend: isFirstForAblation,
                    scalegroup: 'all',
                    points: false,
                    box: { visible: false },
                    meanline: { visible: true },
                    line: { color: ablationColorMap[ablation] },
                    fillcolor: ablationColorMap[ablation],
                    opacity: 0.7,
                    side: 'both',
                    width: violinWidth,
                    hoverinfo: 'skip',
                });
                // Add invisible point for hover
                traces.push({
                    type: 'scatter',
                    mode: 'markers',
                    x: [xPos],
                    y: [successRate],
                    marker: { size: 20, opacity: 0, color: ablationColorMap[ablation] },
                    showlegend: false,
                    hoverinfo: 'text',
                    hovertext: hoverText,
                    hoverlabel: { bgcolor: ablationColorMap[ablation] },
                });
            }
        });
    });

    // Aggregate violins position (after tasks with a gap)
    const aggStartX = task_order.length + 1.0;

    // Compute per-ablation mean as arithmetic average of per-task rates (same as bar chart)
    const ablationMeans = {};
    ablations.forEach(ablation => {
        const rates = [];
        task_order.forEach(task => {
            const taskResults = by_task[task] || [];
            const result = taskResults.find(r => r.ablation === ablation);
            if (result) {
                rates.push(result.success_rate);
            }
        });
        ablationMeans[ablation] = rates.length > 0
            ? rates.reduce((a, b) => a + b, 0) / rates.length
            : 0;
    });

    aggregate.forEach((agg) => {
        const samples = agg.violin_samples;
        if (!samples || samples.length === 0) return;

        const sorted = [...samples].sort((a, b) => a - b);
        const q5 = sorted[Math.floor(sorted.length * 0.05)];
        const q95 = sorted[Math.floor(sorted.length * 0.95)];
        const successRate = ablationMeans[agg.ablation];  // Use arithmetic mean of per-task rates

        const ablIdx = ablations.indexOf(agg.ablation);
        const offset = (ablIdx - (nAbl - 1) / 2) * (groupWidth / nAbl);
        const xPos = aggStartX + offset;

        const hoverText = `${(successRate * 100).toFixed(1)}% [${(q5 * 100).toFixed(0)}-${(q95 * 100).toFixed(0)}]`;
        traces.push({
            type: 'violin',
            y: samples,
            x: samples.map(() => xPos),
            name: agg.ablation,
            legendgroup: agg.ablation,
            showlegend: false,
            scalegroup: 'all',
            points: false,
            box: { visible: false },
            meanline: { visible: true },
            line: { color: ablationColorMap[agg.ablation] },
            fillcolor: ablationColorMap[agg.ablation],
            opacity: 0.7,
            side: 'both',
            width: violinWidth * 1.5,  // Wider for aggregate
            hoverinfo: 'skip',
        });
        // Add invisible point for hover
        traces.push({
            type: 'scatter',
            mode: 'markers',
            x: [xPos],
            y: [successRate],
            marker: { size: 20, opacity: 0, color: ablationColorMap[agg.ablation] },
            showlegend: false,
            hoverinfo: 'text',
            hovertext: hoverText,
            hoverlabel: { bgcolor: ablationColorMap[agg.ablation] },
        });
    });

    // CLD annotations for per-task (only if violin was rendered, deduplicated by ablation)
    const annotations = [];
    task_order.forEach((task, taskIdx) => {
        const taskResults = by_task[task] || [];
        const seenAblations = new Set();
        taskResults.forEach((result) => {
            if (result.cld_letter && result.violin_samples && result.violin_samples.length > 0 && !seenAblations.has(result.ablation)) {
                seenAblations.add(result.ablation);
                const ablIdx = ablations.indexOf(result.ablation);
                const offset = (ablIdx - (nAbl - 1) / 2) * (groupWidth / nAbl);
                annotations.push({
                    x: taskIdx + offset,
                    y: Math.min(result.success_rate + 0.08, 1.02),
                    text: `<b>${result.cld_letter}</b>`,
                    showarrow: false,
                    font: { size: 10, color: 'black' },
                    xref: 'x',
                });
            }
        });
    });

    // CLD annotations for aggregate (only if violin was rendered)
    aggregate.forEach((agg) => {
        if (agg.cld_letter && agg.violin_samples && agg.violin_samples.length > 0) {
            const ablIdx = ablations.indexOf(agg.ablation);
            const offset = (ablIdx - (nAbl - 1) / 2) * (groupWidth / nAbl);
            const successRate = ablationMeans[agg.ablation];  // Use same rate as violin
            annotations.push({
                x: aggStartX + offset,
                y: Math.min(successRate + 0.08, 1.02),
                text: `<b>${agg.cld_letter}</b>`,
                showarrow: false,
                font: { size: 10, color: 'black' },
                xref: 'x',
            });
        }
    });

    // X-axis tick values and labels (truncate long names)
    const maxLabelLen = 25;
    const truncate = (s) => s.length > maxLabelLen ? s.slice(0, maxLabelLen - 2) + '…' : s;
    const tickvals = [...task_order.map((_, i) => i), aggStartX];
    const ticktext = [...task_order.map(truncate), 'Aggregate'];

    const layout = {
        title: 'Success Rate Distribution by Task (Beta Posterior)',
        yaxis: {
            title: 'Success Rate',
            range: [-0.02, 1.15],
            zeroline: false,
        },
        xaxis: {
            title: '',
            tickangle: -45,
            tickvals: tickvals,
            ticktext: ticktext,
            range: [-0.7, aggStartX + 0.7],
        },
        showlegend: true,
        legend: { orientation: 'h', y: 1.02, x: 0.5, xanchor: 'center' },
        annotations: annotations,
        shapes: [
            // Vertical separator before aggregate
            {
                type: 'line',
                x0: task_order.length - 0.15,
                x1: task_order.length - 0.15,
                y0: -0.02,
                y1: 1.1,
                line: { color: 'gray', width: 2 },
            },
        ],
        margin: { b: 150, t: 60, l: 100 },
    };

    Plotly.newPlot('violin-chart', traces, layout, { responsive: true }).then(() => {
        // Trigger resize after render to fix initial sizing
        setTimeout(() => Plotly.Plots.resize('violin-chart'), 100);
    });

    // Clear the separate aggregate chart since we combined them
    document.getElementById('aggregate-violin-chart').innerHTML = '';
}

function renderAggregateViolin(aggregate, ablationColorMap) {
    const traces = [];

    aggregate.forEach((agg, idx) => {
        const samples = agg.violin_samples;
        const sorted = [...samples].sort((a, b) => a - b);
        const q5 = sorted[Math.floor(sorted.length * 0.05)];
        const q95 = sorted[Math.floor(sorted.length * 0.95)];
        const mean = samples.reduce((a, b) => a + b, 0) / samples.length;

        traces.push({
            type: 'violin',
            y: samples,
            name: agg.ablation,
            legendgroup: agg.ablation,
            showlegend: false,
            points: false,
            box: { visible: true },
            meanline: { visible: true },
            line: { color: ablationColorMap[agg.ablation] },
            fillcolor: ablationColorMap[agg.ablation],
            opacity: 0.7,
            x0: agg.ablation,
            hoverinfo: 'text',
            hovertext: `<b>${agg.ablation}</b><br>Mean: ${(mean * 100).toFixed(1)}%<br>Q5: ${(q5 * 100).toFixed(1)}%<br>Q95: ${(q95 * 100).toFixed(1)}%`,
        });
    });

    // Add CLD annotations for aggregate
    const annotations = aggregate.map((agg, idx) => ({
        x: agg.ablation,
        y: Math.min(agg.success_rate + 0.12, 1.02),
        text: `<b>${agg.cld_letter}</b><br>${(agg.success_rate * 100).toFixed(1)}%<br>n=${agg.total}`,
        showarrow: false,
        font: { size: 10 },
    }));

    const layout = {
        title: 'Aggregate',
        yaxis: {
            title: '',
            range: [-0.02, 1.15],
            zeroline: false,
        },
        xaxis: {
            tickangle: -45,
        },
        showlegend: false,
        annotations: annotations,
        shapes: [{
            type: 'line',
            x0: -0.5,
            x1: aggregate.length - 0.5,
            y0: 0.5,
            y1: 0.5,
            line: { color: 'gray', width: 1, dash: 'dash' },
        }],
        margin: { b: 100 },
    };

    Plotly.newPlot('aggregate-violin-chart', traces, layout, { responsive: true });
}

// Beta distribution PDF (using log-gamma for numerical stability)
function logGamma(x) {
    // Lanczos approximation
    const g = 7;
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) {
        return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    }
    x -= 1;
    let a = c[0];
    for (let i = 1; i < g + 2; i++) {
        a += c[i] / (x + i);
    }
    const t = x + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function betaPdf(x, alpha, beta) {
    // Clamp to valid range with small epsilon for numerical stability
    const eps = 1e-10;
    if (x <= eps || x >= 1 - eps) return 0;
    const logB = logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
    const logPdf = (alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - logB;
    return Math.exp(logPdf);
}

function renderScatterPlot(statsData, xAblation, yAblation) {
    const { by_task, task_order } = statsData;

    // Collect data points for scatter
    const points = [];
    const taskColorMap = {};

    task_order.forEach((task, idx) => {
        taskColorMap[task] = VIOLIN_COLORS[idx % VIOLIN_COLORS.length];
        const taskResults = by_task[task] || [];

        const xResult = taskResults.find(r => r.ablation === xAblation);
        const yResult = taskResults.find(r => r.ablation === yAblation);

        if (xResult && yResult) {
            points.push({
                task,
                xRate: xResult.success_rate,
                yRate: yResult.success_rate,
                xAlpha: xResult.alpha,
                xBeta: xResult.beta,
                yAlpha: yResult.alpha,
                yBeta: yResult.beta,
                xTotal: xResult.total,
                yTotal: yResult.total,
                color: taskColorMap[task],
            });
        }
    });

    // Build traces
    const traces = [];

    // Grid for contours (like Python's 200x200)
    const gridSize = 200;
    const gridX = Array.from({ length: gridSize }, (_, i) => 0.001 + i * 0.998 / (gridSize - 1));
    const gridY = Array.from({ length: gridSize }, (_, i) => 0.001 + i * 0.998 / (gridSize - 1));

    // Add 95% confidence regions using Plotly contour (like matplotlib contourf)
    points.forEach((pt) => {
        // Compute joint PDF on grid
        const z = [];
        const flatZ = [];
        for (let j = 0; j < gridSize; j++) {
            const row = [];
            for (let i = 0; i < gridSize; i++) {
                const xPdf = betaPdf(gridX[i], pt.xAlpha, pt.xBeta);
                const yPdf = betaPdf(gridY[j], pt.yAlpha, pt.yBeta);
                const jointPdf = xPdf * yPdf;
                row.push(jointPdf);
                if (jointPdf > 0) flatZ.push(jointPdf);
            }
            z.push(row);
        }

        // Find contour level for 95% confidence (like Python code)
        flatZ.sort((a, b) => b - a);  // Sort descending
        const dx = gridX[1] - gridX[0];
        const dy = gridY[1] - gridY[0];
        const cellArea = dx * dy;

        let cumsum = 0;
        let level = flatZ[0];
        for (let i = 0; i < flatZ.length; i++) {
            cumsum += flatZ[i] * cellArea;
            if (cumsum / (cumsum + (flatZ.length - i - 1) * flatZ[i] * cellArea) >= 0.95) {
                level = flatZ[i];
                break;
            }
        }

        // Normalize cumsum approach (like Python's cumsum / cumsum[-1])
        const totalMass = flatZ.reduce((a, b) => a + b, 0) * cellArea;
        cumsum = 0;
        for (let i = 0; i < flatZ.length; i++) {
            cumsum += flatZ[i] * cellArea;
            if (cumsum / totalMass >= 0.95) {
                level = flatZ[i];
                break;
            }
        }

        // Convert color to rgba for fill
        const rgbaFill = pt.color.startsWith('#')
            ? `rgba(${parseInt(pt.color.slice(1, 3), 16)},${parseInt(pt.color.slice(3, 5), 16)},${parseInt(pt.color.slice(5, 7), 16)},0.3)`
            : pt.color;

        // Use Plotly contour trace (like matplotlib contourf)
        traces.push({
            type: 'contour',
            x: gridX,
            y: gridY,
            z: z,
            contours: {
                start: level,
                end: z.flat().reduce((a, b) => Math.max(a, b), 0),
                size: z.flat().reduce((a, b) => Math.max(a, b), 0) - level + 1,
                coloring: 'fill',
            },
            colorscale: [[0, 'rgba(0,0,0,0)'], [1, rgbaFill]],
            showscale: false,
            hoverinfo: 'skip',
            line: { color: pt.color, width: 1.5 },
            legendgroup: pt.task,
            showlegend: false,
        });
    });

    // Add scatter points on top
    points.forEach(pt => {
        traces.push({
            type: 'scatter',
            mode: 'markers',
            x: [pt.xRate],
            y: [pt.yRate],
            marker: {
                size: 12,
                color: pt.color,
                line: { color: 'white', width: 2 },
            },
            name: pt.task,
            legendgroup: pt.task,
            text: `${pt.task}<br>${xAblation}: ${(pt.xRate * 100).toFixed(1)}%<br>${yAblation}: ${(pt.yRate * 100).toFixed(1)}%`,
            hoverinfo: 'text',
            showlegend: true,
        });
    });

    // Add diagonal line (y = x)
    traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [0, 1],
        y: [0, 1],
        line: { color: 'gray', dash: 'dash', width: 2 },
        showlegend: false,
        hoverinfo: 'skip',
    });

    const layout = {
        title: `${xAblation} vs ${yAblation}`,
        xaxis: {
            title: `${xAblation} Success Rate`,
            range: [0, 1],
            autorange: false,
            zeroline: false,
            constrain: 'domain',
        },
        yaxis: {
            title: `${yAblation} Success Rate`,
            range: [0, 1],
            autorange: false,
            zeroline: false,
            constrain: 'domain',
            scaleanchor: 'x',
            scaleratio: 1,
        },
        showlegend: true,
        legend: { orientation: 'v', x: 1.02, y: 1 },
        hovermode: 'closest',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
    };

    Plotly.newPlot('scatter-chart', traces, layout, { responsive: true, staticPlot: false });
}

function setupScatterControls(statsData) {
    const ablations = [...new Set(statsData.aggregate.map(a => a.ablation))];

    const xSelect = document.getElementById('scatter-x-ablation');
    const ySelect = document.getElementById('scatter-y-ablation');

    xSelect.innerHTML = ablations.map(a => `<option value="${a}">${a}</option>`).join('');
    ySelect.innerHTML = ablations.map(a => `<option value="${a}">${a}</option>`).join('');

    // Set default selections (first two ablations)
    if (ablations.length >= 2) {
        xSelect.value = ablations[0];
        ySelect.value = ablations[1];
    }

    // Add change handlers
    const updateScatter = () => {
        renderScatterPlot(statsData, xSelect.value, ySelect.value);
    };

    xSelect.addEventListener('change', updateScatter);
    ySelect.addEventListener('change', updateScatter);

    // Initial render
    if (ablations.length >= 2) {
        updateScatter();
    }
}

// Tab handling
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;

            // Update button states
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update content visibility
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`tab-${tabId}`).classList.add('active');

            // Trigger Plotly resize for charts that may need it
            setTimeout(() => {
                if (tabId === 'violin' && document.getElementById('violin-chart').data) {
                    Plotly.Plots.resize('violin-chart');
                }
                if (tabId === 'scatter' && document.getElementById('scatter-chart').data) {
                    Plotly.Plots.resize('scatter-chart');
                }
            }, 50);
        });
    });
}

// Filter data for display (exclude reference-only rows)
function filterDisplayData(data) {
    if (selectedAblations.size === 0) return data;
    return data.filter(row => selectedAblations.has(row.ablation));
}

async function refresh() {
    elements.leaderboardBody.innerHTML = `
        <tr><td colspan="8" class="loading">Loading...</td></tr>
    `;

    // Update URL to reflect current filter state
    updateUrlParams();

    try {
        const [data, totalStats] = await Promise.all([
            fetchLeaderboard(),
            fetchStats(),
        ]);

        // Update task/ablation chips based on available data for selected campaigns
        if (selectedCampaigns.size > 0) {
            const availableTasks = new Set(data.map(d => d.task_name));
            const availableAblations = new Set(data.map(d => d.ablation).filter(Boolean));

            // Deselect any items that are no longer available
            selectedTasks.forEach(t => { if (!availableTasks.has(t)) selectedTasks.delete(t); });
            selectedAblations.forEach(a => { if (!availableAblations.has(a)) selectedAblations.delete(a); });

            renderTaskChips(masterTasks, availableTasks);
            renderAblationChips(masterAblations, availableAblations);
            populateReferenceDatalist(masterAblations, availableAblations);
        } else {
            renderTaskChips(masterTasks);
            renderAblationChips(masterAblations);
            populateReferenceDatalist(masterAblations);
        }

        // Filter for display but keep full data for reference lookup
        const displayData = filterDisplayData(data);
        renderLeaderboard(data, displayData);
        updateStats(displayData, totalStats);
    } catch (error) {
        elements.leaderboardBody.innerHTML = `
            <tr><td colspan="8" class="loading">Error: ${error.message}</td></tr>
        `;
    }
}

async function init() {
    // Load filters and config schema
    try {
        const [filters, schema] = await Promise.all([
            fetchFilters(),
            fetchConfigSchema(),
        ]);
        console.log('Filters loaded:', filters);
        console.log('Schema loaded:', Object.keys(schema).length, 'fields');
        console.log('Ablations:', filters.ablations);

        // Store master lists from database
        masterCampaigns = filters.campaigns.map(c => c.name);
        masterTasks = filters.tasks;
        masterAblations = filters.ablations;

        // Apply URL params and render chips
        applyUrlParams();
        renderCampaignChips(masterCampaigns);

        configSchema = schema;
    } catch (error) {
        console.error('Failed to load filters:', error);
    }

    // Event listeners
    elements.refreshBtn.addEventListener('click', refresh);
    elements.compareBtn.addEventListener('click', handleCompare);
    elements.exportBtn.addEventListener('click', handleExport);
    elements.selectAll.addEventListener('change', handleSelectAll);
    elements.addConfigFilterBtn.addEventListener('click', addConfigFilterRow);

    // Campaign All/Clear buttons
    document.getElementById('campaign-all-btn').addEventListener('click', selectAllCampaigns);
    document.getElementById('campaign-clear-btn').addEventListener('click', clearAllCampaigns);

    // Task All/Clear buttons
    document.getElementById('task-all-btn').addEventListener('click', selectAllTasks);
    document.getElementById('task-clear-btn').addEventListener('click', clearAllTasks);

    // Ablation All/Clear buttons
    document.getElementById('ablation-all-btn').addEventListener('click', selectAllAblations);
    document.getElementById('ablation-clear-btn').addEventListener('click', clearAllAblations);

    // Campaign search
    elements.campaignSearch.addEventListener('input', (e) => {
        filterCampaignChips(e.target.value);
    });

    // Task search
    elements.taskSearch.addEventListener('input', (e) => {
        filterTaskChips(e.target.value);
    });

    // Ablation search
    elements.ablationSearch.addEventListener('input', (e) => {
        filterAblationChips(e.target.value);
    });

    // Reference select (searchable datalist)
    elements.referenceSelect.addEventListener('change', (e) => {
        const value = e.target.value;
        // Only set if the value exists in the datalist
        if (allReferences.includes(value)) {
            selectedReference = value;
            refresh();
        } else if (value === '') {
            selectedReference = null;
            refresh();
        }
    });
    // Also handle input event for immediate feedback when selecting from datalist
    elements.referenceSelect.addEventListener('input', (e) => {
        const value = e.target.value;
        if (allReferences.includes(value)) {
            selectedReference = value;
            refresh();
        } else if (value === '') {
            selectedReference = null;
            refresh();
        }
    });

    // Initialize tabs
    initTabs();

    elements.clearSelectionBtn.addEventListener('click', () => {
        selectedRows.clear();
        document.querySelectorAll('.row-checkbox').forEach(cb => cb.checked = false);
        elements.selectAll.checked = false;
        updateSelectionUI();
        updateUrlParams();
    });

    elements.closeCompare.addEventListener('click', () => {
        elements.compareModal.classList.add('hidden');
    });

    elements.closeConfig.addEventListener('click', () => {
        elements.configModal.classList.add('hidden');
    });

    // Close modals on outside click
    elements.compareModal.addEventListener('click', (e) => {
        if (e.target === elements.compareModal) {
            elements.compareModal.classList.add('hidden');
        }
    });

    elements.configModal.addEventListener('click', (e) => {
        if (e.target === elements.configModal) {
            elements.configModal.classList.add('hidden');
        }
    });

    // Sort handlers
    document.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', handleSort);
    });

    // Filter change handlers
    elements.minRollouts.addEventListener('change', refresh);

    // Initial load
    updateSortIndicators();
    refresh();
}

// URL Parameter Functions for Shareable Links
function getUrlParams() {
    const params = new URLSearchParams(window.location.search);
    return {
        campaigns: params.getAll('campaign'),
        tasks: params.getAll('task'),
        ablations: params.getAll('ablation'),
        reference: params.get('reference'),
        minRollouts: params.get('minRollouts'),
        sortColumn: params.get('sort'),
        sortDesc: params.get('desc'),
        selected: params.getAll('selected'),
    };
}

function updateUrlParams() {
    const params = new URLSearchParams();

    // Campaigns
    selectedCampaigns.forEach(c => params.append('campaign', c));

    // Tasks
    selectedTasks.forEach(t => params.append('task', t));

    // Ablations
    selectedAblations.forEach(a => params.append('ablation', a));

    // Reference
    if (selectedReference) {
        params.set('reference', selectedReference);
    }

    // Min rollouts
    if (elements.minRollouts.value) {
        params.set('minRollouts', elements.minRollouts.value);
    }

    // Sort
    if (currentSort.column !== 'success_rate' || !currentSort.descending) {
        params.set('sort', currentSort.column);
        params.set('desc', currentSort.descending);
    }

    // Selected rows (for comparison)
    selectedRows.forEach(rowKey => params.append('selected', rowKey));

    // Update URL without reloading
    const newUrl = params.toString()
        ? `${window.location.pathname}?${params.toString()}`
        : window.location.pathname;
    window.history.replaceState({}, '', newUrl);
}

function applyUrlParams() {
    const params = getUrlParams();

    // Apply campaigns
    if (params.campaigns.length > 0) {
        params.campaigns.forEach(c => selectedCampaigns.add(c));
    }

    // Apply tasks
    if (params.tasks.length > 0) {
        params.tasks.forEach(t => selectedTasks.add(t));
    }

    // Apply ablations
    if (params.ablations.length > 0) {
        params.ablations.forEach(a => selectedAblations.add(a));
    }

    // Apply reference
    if (params.reference) {
        selectedReference = params.reference;
        elements.referenceSelect.value = params.reference;
    }

    // Apply min rollouts
    if (params.minRollouts) {
        elements.minRollouts.value = params.minRollouts;
    }

    // Apply sort
    if (params.sortColumn) {
        currentSort.column = params.sortColumn;
        currentSort.descending = params.sortDesc === 'true';
    }

    // Apply selected rows
    if (params.selected.length > 0) {
        params.selected.forEach(rowKey => selectedRows.add(rowKey));
    }
}

// Start
document.addEventListener('DOMContentLoaded', init);
