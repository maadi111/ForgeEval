/**
 * ForgeEval — Application Controller & UI Logic
 * Production-grade ML evaluation platform client.
 */

(function () {
  'use strict';

  const DATA = window.FORGEEVAL_DATA || {};
  let currentView = 'overview';
  let currentBenchmark = DATA.benchmarks ? DATA.benchmarks[0] : null;
  let currentRun = DATA.runs ? DATA.runs[0] : null;
  let currentAgent = DATA.agents ? DATA.agents[0] : null;
  let wizardStep = 1;
  let activeState = 'normal'; // 'normal', 'skeleton', 'empty', 'error'

  // =========================================================================
  // Initialization & Live Health Polling
  // =========================================================================

  document.addEventListener('DOMContentLoaded', () => {
    initRouter();
    initCommandPalette();
    initModals();
    initStateSwitcher();
    pollLiveApiHealth();
    renderAllViews();
  });

  async function pollLiveApiHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const dot = document.querySelector('.status-dot');
        const text = document.getElementById('sidebar-status-text');
        if (dot) dot.className = 'status-dot';
        if (text) text.textContent = 'API & Runner Online';
      }
    } catch {
      // Offline fallback: keep default telemetry
    }
  }

  // =========================================================================
  // Router & View Management
  // =========================================================================

  function initRouter() {
    // Sidebar nav clicks
    document.querySelectorAll('.nav-item').forEach(item => {
      item.addEventListener('click', () => {
        const targetView = item.getAttribute('data-view');
        if (targetView) navigateTo(targetView);
      });
    });

    // In-page quick jumps
    document.addEventListener('click', (e) => {
      const jumpEl = e.target.closest('[data-jump]');
      if (jumpEl) {
        const target = jumpEl.getAttribute('data-jump');
        navigateTo(target);
      }
    });

    // Topbar Primary Action context
    const topbarPrimary = document.getElementById('btn-topbar-primary');
    if (topbarPrimary) {
      topbarPrimary.addEventListener('click', () => {
        if (currentView === 'benchmarks') {
          navigateTo('create-benchmark');
        } else {
          openEvalModal();
        }
      });
    }

    // Benchmark sub-tabs
    document.querySelectorAll('.sub-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const parent = btn.parentElement;
        parent.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const subtabId = btn.getAttribute('data-subtab');
        document.querySelectorAll('.subtab-pane').forEach(p => p.style.display = 'none');
        const targetPane = document.getElementById(`subtab-${subtabId}`);
        if (targetPane) targetPane.style.display = 'block';
      });
    });

    // Wizard navigation
    const btnWizardNext = document.getElementById('btn-wizard-next');
    if (btnWizardNext) {
      btnWizardNext.addEventListener('click', handleWizardNext);
    }
    const btnCancelWizard = document.getElementById('btn-cancel-create-wizard');
    if (btnCancelWizard) {
      btnCancelWizard.addEventListener('click', () => navigateTo('benchmarks'));
    }
    const btnFinishPublish = document.getElementById('btn-finish-publish');
    if (btnFinishPublish) {
      btnFinishPublish.addEventListener('click', handlePublishBenchmark);
    }

    // Quick filters
    const searchBenchmarks = document.getElementById('benchmark-search-input');
    if (searchBenchmarks) {
      searchBenchmarks.addEventListener('input', () => renderBenchmarksList());
    }
    const catFilter = document.getElementById('benchmark-category-filter');
    if (catFilter) {
      catFilter.addEventListener('change', () => renderBenchmarksList());
    }
    const diffFilter = document.getElementById('benchmark-difficulty-filter');
    if (diffFilter) {
      diffFilter.addEventListener('change', () => renderBenchmarksList());
    }

    const searchRuns = document.getElementById('run-search-input');
    if (searchRuns) {
      searchRuns.addEventListener('input', () => renderRunsList());
    }
    const runStatusFilter = document.getElementById('run-status-filter');
    if (runStatusFilter) {
      runStatusFilter.addEventListener('change', () => renderRunsList());
    }

    const btnBackRuns = document.getElementById('btn-back-to-runs');
    if (btnBackRuns) {
      btnBackRuns.addEventListener('click', () => navigateTo('runs'));
    }

    const btnCompareCTA = document.getElementById('btn-compare-runs-cta');
    if (btnCompareCTA) {
      btnCompareCTA.addEventListener('click', () => navigateTo('compare'));
    }

    const btnCopyLogs = document.getElementById('btn-copy-logs');
    if (btnCopyLogs) {
      btnCopyLogs.addEventListener('click', () => {
        showToast('Container logs copied to clipboard', 'info');
      });
    }

    // Create Benchmark CTA
    const btnCreateBenchModal = document.getElementById('btn-create-benchmark-modal');
    if (btnCreateBenchModal) {
      btnCreateBenchModal.addEventListener('click', () => navigateTo('create-benchmark'));
    }
  }

  function navigateTo(viewId, params = {}) {
    currentView = viewId;

    // Update active nav item
    document.querySelectorAll('.nav-item').forEach(item => {
      item.classList.toggle('active', item.getAttribute('data-view') === viewId);
    });

    // Update active view pane
    document.querySelectorAll('.view-pane').forEach(pane => {
      pane.classList.remove('active');
    });
    const targetPane = document.getElementById(`view-${viewId}`);
    if (targetPane) {
      targetPane.classList.add('active');
    }

    // Update topbar context
    updateTopbar(viewId, params);

    // Refresh specific view content
    if (viewId === 'benchmark-detail') {
      if (params.id) {
        currentBenchmark = DATA.benchmarks.find(b => b.id === params.id) || DATA.benchmarks[0];
      }
      renderBenchmarkDetail();
    } else if (viewId === 'run-detail') {
      if (params.id) {
        currentRun = DATA.runs.find(r => r.id === params.id) || DATA.runs[0];
      }
      renderRunDetail();
    } else if (viewId === 'agent-detail') {
      if (params.id) {
        currentAgent = DATA.agents.find(a => a.id === params.id) || DATA.agents[0];
      }
      renderAgentDetail();
    }
  }

  function updateTopbar(viewId, params = {}) {
    const breadcrumbPage = document.getElementById('breadcrumb-active-page');
    const primaryBtnText = document.getElementById('topbar-primary-text');

    const titles = {
      'overview': 'Overview',
      'benchmarks': 'Benchmarks',
      'benchmark-detail': currentBenchmark ? currentBenchmark.id : 'Benchmark Detail',
      'create-benchmark': 'Create Benchmark',
      'runs': 'Runs',
      'run-detail': currentRun ? `Run ${currentRun.id}` : 'Run Detail',
      'compare': 'Run Comparison',
      'agents': 'Agents',
      'agent-detail': currentAgent ? currentAgent.id : 'Agent Detail',
      'graders': 'Graders & Safety',
      'artifacts': 'Artifacts',
      'environments': 'Environments',
      'analytics': 'Analytics',
      'system-health': 'System Health',
      'settings': 'Settings'
    };

    if (breadcrumbPage) {
      breadcrumbPage.textContent = titles[viewId] || 'Overview';
    }

    if (primaryBtnText) {
      if (viewId === 'benchmarks') {
        primaryBtnText.textContent = 'Create Benchmark';
      } else if (viewId === 'runs' || viewId === 'overview') {
        primaryBtnText.textContent = 'Run Evaluation';
      } else {
        primaryBtnText.textContent = 'Run Evaluation';
      }
    }
  }

  // =========================================================================
  // View Renderers
  // =========================================================================

  function renderAllViews() {
    renderOverview();
    renderBenchmarksList();
    renderBenchmarkDetail();
    renderRunsList();
    renderAgentsList();
    renderGraders();
    renderArtifacts();
    renderEnvironments();
    renderAnalytics();
    renderSystemHealth();
    renderSettings();
  }

  function renderOverview() {
    // Failure Modes List
    const failureList = document.getElementById('failure-modes-list');
    if (failureList && DATA.analytics && DATA.analytics.failure_modes) {
      failureList.innerHTML = DATA.analytics.failure_modes.map(mode => `
        <div class="breakdown-row">
          <div class="breakdown-meta">
            <span class="breakdown-name">${mode.name}</span>
            <span class="breakdown-val">${mode.count} (${mode.pct})</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill ${mode.count > 10 ? 'danger' : 'warning'}" style="width: ${mode.pct};"></div>
          </div>
        </div>
      `).join('');
    }

    // Recent Runs
    const tbodyRecent = document.getElementById('tbody-recent-runs');
    if (tbodyRecent && DATA.runs) {
      tbodyRecent.innerHTML = DATA.runs.slice(0, 5).map(r => `
        <tr onclick="window.ForgeEvalApp.openRun('${r.id}')">
          <td class="table-mono">${r.id}</td>
          <td><strong>${r.benchmark_name}</strong></td>
          <td><span class="badge badge-neutral">${r.agent}</span></td>
          <td class="table-mono">${r.commit}</td>
          <td><strong style="color: ${r.score >= 90 ? 'var(--success-text)' : 'var(--danger-text)'};">${r.score.toFixed(1)} / 100</strong></td>
          <td class="table-mono">${r.duration}</td>
          <td><span class="badge ${r.status === 'PASSED' ? 'badge-pass' : 'badge-fail'}">${r.status}</span></td>
        </tr>
      `).join('');
    }
  }

  function renderBenchmarksList() {
    const tbody = document.getElementById('tbody-benchmarks-index');
    if (!tbody || !DATA.benchmarks) return;

    const searchTerm = (document.getElementById('benchmark-search-input')?.value || '').toLowerCase();
    const category = document.getElementById('benchmark-category-filter')?.value || 'all';
    const difficulty = document.getElementById('benchmark-difficulty-filter')?.value || 'all';

    const filtered = DATA.benchmarks.filter(b => {
      const matchSearch = b.name.toLowerCase().includes(searchTerm) || b.id.toLowerCase().includes(searchTerm) || b.tags.some(t => t.toLowerCase().includes(searchTerm));
      const matchCat = category === 'all' || b.category === category;
      const matchDiff = difficulty === 'all' || b.difficulty === difficulty;
      return matchSearch && matchCat && matchDiff;
    });

    const counter = document.getElementById('benchmarks-counter-text');
    if (counter) counter.textContent = `Showing ${filtered.length} of ${DATA.benchmarks.length} benchmarks`;

    if (filtered.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 30px; color: var(--text-tertiary);">No benchmarks found matching filter criteria.</td></tr>`;
      return;
    }

    tbody.innerHTML = filtered.map(b => `
      <tr onclick="window.ForgeEvalApp.openBenchmark('${b.id}')">
        <td class="table-mono">${b.id}</td>
        <td><strong>${b.name}</strong></td>
        <td><span class="badge badge-neutral">${b.category}</span></td>
        <td><span class="badge ${b.difficulty === 'Hard' ? 'badge-info' : 'badge-neutral'}">${b.difficulty}</span></td>
        <td class="table-mono">${b.version}</td>
        <td>${b.test_summary.public} pub / ${b.test_summary.hidden} hid / ${b.test_summary.negative_controls} neg</td>
        <td class="table-mono">${b.avg_runtime}</td>
        <td><strong>${b.pass_rate}</strong></td>
        <td><span class="badge badge-pass">${b.status}</span></td>
      </tr>
    `).join('');
  }

  function renderBenchmarkDetail() {
    const b = currentBenchmark || DATA.benchmarks[0];
    if (!b) return;

    // Badges & Headers
    const idBadge = document.getElementById('detail-badge-id');
    if (idBadge) idBadge.textContent = b.id;
    const diffBadge = document.getElementById('detail-badge-difficulty');
    if (diffBadge) diffBadge.textContent = b.difficulty;
    const fwBadge = document.getElementById('detail-badge-framework');
    if (fwBadge) fwBadge.textContent = b.framework;
    const titleEl = document.getElementById('detail-heading-title');
    if (titleEl) titleEl.textContent = b.name;
    const descEl = document.getElementById('detail-heading-desc');
    if (descEl) descEl.textContent = b.description;

    // Overview Tab fields
    const symEl = document.getElementById('b-detail-symptoms');
    if (symEl) symEl.textContent = b.known_symptoms;
    const expEl = document.getElementById('b-detail-expected');
    if (expEl) expEl.textContent = b.expected_behavior;
    const solveEl = document.getElementById('b-detail-solve-time');
    if (solveEl) solveEl.textContent = b.estimated_solve_time;
    const envEl = document.getElementById('b-detail-env');
    if (envEl) envEl.textContent = b.environment_reqs ? `${b.environment_reqs.cpu}, ${b.environment_reqs.memory}` : 'Docker';
    const slaEl = document.getElementById('b-detail-sla');
    if (slaEl) slaEl.textContent = b.runtime_sla;
    const datasetEl = document.getElementById('b-detail-dataset');
    if (datasetEl) datasetEl.textContent = b.dataset_info ? `${b.dataset_info.name} (${b.dataset_info.size})` : 'Parquet dataset';

    // Skills
    const skillsContainer = document.getElementById('b-detail-skills');
    if (skillsContainer && b.required_skills) {
      skillsContainer.innerHTML = b.required_skills.map(s => `<span class="badge badge-neutral">${s}</span>`).join('');
    }

    // Workspace Files Tree
    const treeContainer = document.getElementById('workspace-tree-items');
    if (treeContainer && b.files) {
      treeContainer.innerHTML = b.files.map(f => `
        <div class="tree-item ${f.active ? 'active' : ''}" onclick="window.ForgeEvalApp.selectWorkspaceFile('${f.path}')">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          <span>${f.name}</span>
        </div>
      `).join('');
    }

    // Tests Tab
    const tbodyTests = document.getElementById('tbody-benchmark-tests');
    if (tbodyTests && b.tests) {
      tbodyTests.innerHTML = b.tests.map(t => `
        <tr>
          <td class="table-mono">${t.id}</td>
          <td>${t.name}</td>
          <td><span class="badge badge-neutral">${t.suite}</span></td>
          <td><span class="badge badge-info">${t.type}</span></td>
          <td class="table-mono">${t.runtime}</td>
          <td><span class="badge ${t.status === 'PASS' ? 'badge-pass' : 'badge-fail'}">${t.status}</span></td>
        </tr>
      `).join('');
    }

    // Grader Weights
    const weightsContainer = document.getElementById('grader-weights-breakdown');
    if (weightsContainer && b.weights) {
      weightsContainer.innerHTML = Object.entries(b.weights).map(([k, v]) => `
        <div class="breakdown-row">
          <div class="breakdown-meta">
            <span class="breakdown-name" style="text-transform: capitalize;">${k.replace('_', ' ')}</span>
            <span class="breakdown-val">${v} pts</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" style="width: ${v * 2.5}%;"></div>
          </div>
        </div>
      `).join('');
    }

    // Reference Solutions
    const refContainer = document.getElementById('reference-solutions-container');
    if (refContainer && b.sol_a && b.sol_b) {
      refContainer.innerHTML = `
        <div class="panel-card" style="padding: 14px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h4 style="font-size: 13px; font-weight: 600; color: #93c5fd;">${b.sol_a.name}</h4>
            <span class="badge badge-pass">${b.sol_a.status}</span>
          </div>
          <p style="font-size: 12px; color: var(--text-secondary); margin-bottom: 10px;">${b.sol_a.approach}</p>
          <div style="font-size: 11px; font-family: var(--font-mono); color: var(--text-tertiary); display: flex; gap: 12px;">
            <span>Commit: ${b.sol_a.commit}</span>
            <span>Runtime: ${b.sol_a.runtime}</span>
            <span>Memory: ${b.sol_a.memory}</span>
          </div>
        </div>
        <div class="panel-card" style="padding: 14px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h4 style="font-size: 13px; font-weight: 600; color: #93c5fd;">${b.sol_b.name}</h4>
            <span class="badge badge-pass">${b.sol_b.status}</span>
          </div>
          <p style="font-size: 12px; color: var(--text-secondary); margin-bottom: 10px;">${b.sol_b.approach}</p>
          <div style="font-size: 11px; font-family: var(--font-mono); color: var(--text-tertiary); display: flex; gap: 12px;">
            <span>Commit: ${b.sol_b.commit}</span>
            <span>Runtime: ${b.sol_b.runtime}</span>
            <span>Memory: ${b.sol_b.memory}</span>
          </div>
        </div>
      `;
    }

    // Artifacts Tab
    const tbodyArtifacts = document.getElementById('tbody-benchmark-artifacts');
    if (tbodyArtifacts && DATA.artifacts) {
      const filteredArts = DATA.artifacts.filter(a => a.used_by.includes(b.id) || a.used_by.includes('Global') || a.used_by.includes('All'));
      tbodyArtifacts.innerHTML = filteredArts.map(a => `
        <tr>
          <td><strong>${a.name}</strong></td>
          <td><span class="badge badge-neutral">${a.type}</span></td>
          <td class="table-mono">${a.version}</td>
          <td class="table-mono">${a.hash}</td>
          <td class="table-mono">${a.size}</td>
          <td>${a.created}</td>
        </tr>
      `).join('');
    }

    // Runs Tab
    const tbodyRuns = document.getElementById('tbody-benchmark-runs');
    if (tbodyRuns && DATA.runs) {
      const bRuns = DATA.runs.filter(r => r.benchmark_id === b.id);
      tbodyRuns.innerHTML = bRuns.map(r => `
        <tr onclick="window.ForgeEvalApp.openRun('${r.id}')">
          <td class="table-mono">${r.id}</td>
          <td><span class="badge badge-neutral">${r.agent}</span></td>
          <td class="table-mono">${r.commit}</td>
          <td>${r.started_at}</td>
          <td class="table-mono">${r.duration}</td>
          <td><strong>${r.score.toFixed(1)} / 100</strong></td>
          <td><span class="badge ${r.status === 'PASSED' ? 'badge-pass' : 'badge-fail'}">${r.status}</span></td>
        </tr>
      `).join('');
    }
  }

  function renderRunsList() {
    const tbody = document.getElementById('tbody-runs-global');
    if (!tbody || !DATA.runs) return;

    const searchTerm = (document.getElementById('run-search-input')?.value || '').toLowerCase();
    const status = document.getElementById('run-status-filter')?.value || 'all';

    const filtered = DATA.runs.filter(r => {
      const matchSearch = r.id.toLowerCase().includes(searchTerm) || r.agent.toLowerCase().includes(searchTerm) || r.benchmark_name.toLowerCase().includes(searchTerm);
      const matchStatus = status === 'all' || r.status === status;
      return matchSearch && matchStatus;
    });

    tbody.innerHTML = filtered.map(r => `
      <tr onclick="window.ForgeEvalApp.openRun('${r.id}')">
        <td class="table-mono">${r.id}</td>
        <td><strong>${r.benchmark_name}</strong></td>
        <td><span class="badge badge-neutral">${r.agent}</span></td>
        <td class="table-mono">${r.commit}</td>
        <td>${r.started_at}</td>
        <td class="table-mono">${r.duration}</td>
        <td><strong style="color: ${r.score >= 90 ? 'var(--success-text)' : 'var(--danger-text)'};">${r.score.toFixed(1)} / 100</strong></td>
        <td><span class="badge ${r.status === 'PASSED' ? 'badge-pass' : 'badge-fail'}">${r.status}</span></td>
      </tr>
    `).join('');
  }

  function renderRunDetail() {
    const r = currentRun || DATA.runs[0];
    if (!r) return;

    document.getElementById('rd-badge-id').textContent = r.id;
    const statusBadge = document.getElementById('rd-badge-status');
    statusBadge.textContent = r.status;
    statusBadge.className = `badge ${r.status === 'PASSED' ? 'badge-pass' : 'badge-fail'}`;

    document.getElementById('rd-badge-agent').textContent = r.agent;
    document.getElementById('rd-heading-title').textContent = `Run ${r.id} • ${r.benchmark_name}`;
    document.getElementById('rd-heading-meta').textContent = `Commit ${r.commit} • Duration ${r.duration} • Environment: ${r.environment}`;
    document.getElementById('rd-total-time').textContent = r.duration;
    document.getElementById('rd-score-badge').textContent = `${r.score.toFixed(1)} / 100`;

    // Failure Analysis (Section 23)
    const failContainer = document.getElementById('rd-failure-container');
    if (failContainer) {
      if (r.failure_analysis) {
        const fa = r.failure_analysis;
        failContainer.innerHTML = `
          <div class="failure-analysis-card">
            <div class="failure-banner">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              <span>Failure Analysis: ${fa.summary}</span>
            </div>
            <div class="failure-grid">
              <div class="failure-block">
                <h4>Failed Test</h4>
                <p><span class="badge badge-fail">${fa.failed_test}</span></p>
              </div>
              <div class="failure-block">
                <h4>Observed Result</h4>
                <p>${fa.observed}</p>
              </div>
              <div class="failure-block">
                <h4>Expected Behavior</h4>
                <p>${fa.expected}</p>
              </div>
              <div class="failure-block">
                <h4>Evidence & Investigation Hint</h4>
                <p>${fa.suggested_investigation}</p>
              </div>
            </div>
          </div>
        `;
      } else {
        failContainer.innerHTML = '';
      }
    }

    // Timeline (Section 21)
    const timelineList = document.getElementById('rd-timeline-list');
    if (timelineList && r.stages) {
      timelineList.innerHTML = r.stages.map(st => `
        <div class="timeline-item">
          <div class="timeline-node ${st.status === 'OK' ? 'pass' : 'fail'}"></div>
          <div class="timeline-title-row">
            <span class="timeline-step-name">${st.name}</span>
            <span class="timeline-step-time">${st.duration || ''}</span>
          </div>
          <span class="timeline-detail">${st.detail || 'Executed inside isolated sandbox container.'}</span>
        </div>
      `).join('');
    }

    // Score Breakdown (Section 18)
    const scoreList = document.getElementById('rd-score-breakdown-list');
    if (scoreList && r.score_breakdown) {
      scoreList.innerHTML = Object.entries(r.score_breakdown).filter(([k]) => k !== 'total').map(([k, v]) => `
        <div class="breakdown-row">
          <div class="breakdown-meta">
            <span class="breakdown-name" style="text-transform: capitalize;">${k.replace('_', ' ')}</span>
            <span class="breakdown-val">${v.toFixed(1)} pts</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill ${v > 0 ? 'success' : 'danger'}" style="width: ${Math.min(100, v * 2.5)}%;"></div>
          </div>
        </div>
      `).join('');
    }

    // Code Diff (Section 24)
    const diffBody = document.getElementById('rd-diff-body');
    if (diffBody && r.diff) {
      const lines = r.diff.patch.split('\n');
      diffBody.innerHTML = lines.map(line => {
        let cls = 'diff-line';
        if (line.startsWith('+') && !line.startsWith('+++')) cls += ' diff-add';
        else if (line.startsWith('-') && !line.startsWith('---')) cls += ' diff-del';
        else if (line.startsWith('@@')) cls += ' diff-meta';
        return `<div class="${cls}">${escapeHtml(line)}</div>`;
      }).join('');
    }

    // Terminal Logs (Section 22)
    const termBody = document.getElementById('rd-terminal-body');
    if (termBody && r.logs) {
      termBody.innerHTML = r.logs.map(line => {
        let cls = '';
        if (line.includes('INFO')) cls = 'log-info';
        else if (line.includes('WARN')) cls = 'log-warn';
        else if (line.includes('ERROR')) cls = 'log-err';
        else if (line.includes('PASS') || line.includes('PASSED')) cls = 'log-ok';
        return `<div class="${cls}">${escapeHtml(line)}</div>`;
      }).join('');
    }
  }

  function renderAgentsList() {
    const tbody = document.getElementById('tbody-agents-list');
    if (!tbody || !DATA.agents) return;

    tbody.innerHTML = DATA.agents.map(a => `
      <tr onclick="window.ForgeEvalApp.openAgent('${a.id}')">
        <td><strong>${a.name}</strong></td>
        <td><span class="badge badge-neutral">${a.type}</span></td>
        <td class="table-mono">${a.runs}</td>
        <td><strong>${a.pass_rate}</strong></td>
        <td><strong style="color: var(--success-text);">${a.avg_score}</strong></td>
        <td class="table-mono">${a.median_runtime}</td>
        <td><span class="badge ${parseFloat(a.failure_rate) > 20 ? 'badge-warn' : 'badge-neutral'}">${a.failure_rate}</span></td>
        <td><span class="badge badge-pass">${a.benchmark_coverage}</span></td>
      </tr>
    `).join('');
  }

  function renderAgentDetail() {
    const a = currentAgent || DATA.agents[0];
    if (!a) return;

    document.getElementById('ad-name').textContent = a.name;
    document.getElementById('ad-type').textContent = `${a.type} • ${a.provider}`;
    document.getElementById('ad-runs').textContent = a.runs;
    document.getElementById('ad-pass-rate').textContent = a.pass_rate;
    document.getElementById('ad-score').textContent = a.avg_score;
    document.getElementById('ad-runtime').textContent = a.median_runtime;
    document.getElementById('ad-timeout').textContent = (a.failures_by_category && a.failures_by_category.Timeout) ? `${a.failures_by_category.Timeout} runs` : '0';

    // Agent Trace (Section 27)
    const traceList = document.getElementById('ad-trace-list');
    if (traceList) {
      traceList.innerHTML = `
        <div class="timeline-item">
          <div class="timeline-node pass"></div>
          <div class="timeline-title-row">
            <span class="timeline-step-name">1. Task Received & Workspace Mounted</span>
            <span class="timeline-step-time">00:00</span>
          </div>
          <span class="timeline-detail">Agent received problem description and cloned target repository in isolated container.</span>
        </div>
        <div class="timeline-item">
          <div class="timeline-node pass"></div>
          <div class="timeline-title-row">
            <span class="timeline-step-name">2. Repository Inspected (Code Analysis)</span>
            <span class="timeline-step-time">00:24</span>
          </div>
          <span class="timeline-detail">Scanned features.py, train.py, predict.py; analyzed rolling window calculations and temporal ordering.</span>
        </div>
        <div class="timeline-item">
          <div class="timeline-node pass"></div>
          <div class="timeline-title-row">
            <span class="timeline-step-name">3. Public Tests Executed</span>
            <span class="timeline-step-time">00:48</span>
          </div>
          <span class="timeline-detail">Executed pytest tests/public. 2/2 tests passed (noting public tests do not test for temporal lookahead).</span>
        </div>
        <div class="timeline-item">
          <div class="timeline-node pass"></div>
          <div class="timeline-title-row">
            <span class="timeline-step-name">4. Code Patch Applied</span>
            <span class="timeline-step-time">01:32</span>
          </div>
          <span class="timeline-detail">Refactored rolling aggregations to apply .shift(1) offset, isolating historical timestamps from prediction cutoffs.</span>
        </div>
        <div class="timeline-item">
          <div class="timeline-node pass"></div>
          <div class="timeline-title-row">
            <span class="timeline-step-name">5. Re-test & Verification</span>
            <span class="timeline-step-time">02:10</span>
          </div>
          <span class="timeline-detail">Re-executed contract test suite. Schema invariance and non-null guarantees verified.</span>
        </div>
        <div class="timeline-item">
          <div class="timeline-node pass"></div>
          <div class="timeline-title-row">
            <span class="timeline-step-name">6. Final Submission & Behavioral Evaluation</span>
            <span class="timeline-step-time">03:42</span>
          </div>
          <span class="timeline-detail">Submitted patch to ForgeEval Grader. All 18 hidden behavioral tests and 6 negative controls passed. Awarded 100/100.</span>
        </div>
      `;
    }
  }

  function renderGraders() {
    const tbody = document.getElementById('tbody-negative-controls');
    if (!tbody || !DATA.grader_health || !DATA.grader_health.active_negative_controls) return;

    tbody.innerHTML = DATA.grader_health.active_negative_controls.map(nc => `
      <tr>
        <td class="table-mono">${nc.id}</td>
        <td><strong>${nc.target}</strong></td>
        <td>${nc.name}</td>
        <td><span class="badge badge-pass">${nc.status}</span></td>
        <td class="table-mono">${nc.last_verified}</td>
      </tr>
    `).join('');
  }

  function renderArtifacts() {
    const tbody = document.getElementById('tbody-artifacts-global');
    if (!tbody || !DATA.artifacts) return;

    tbody.innerHTML = DATA.artifacts.map(a => `
      <tr>
        <td><strong>${a.name}</strong></td>
        <td><span class="badge badge-neutral">${a.type}</span></td>
        <td class="table-mono">${a.version}</td>
        <td class="table-mono">${a.hash}</td>
        <td class="table-mono">${a.size}</td>
        <td>${a.created}</td>
        <td><span class="badge badge-info">${a.used_by}</span></td>
      </tr>
    `).join('');
  }

  function renderEnvironments() {
    const container = document.getElementById('environments-cards-grid');
    if (!container || !DATA.environments) return;

    container.innerHTML = DATA.environments.map(env => `
      <div class="panel-card" style="padding: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <h4 style="font-size: 13px; font-weight: 600; color: #93c5fd;">${env.name}</h4>
          <span class="badge badge-pass">${env.reproducibility}</span>
        </div>
        <p style="font-size: 11.5px; color: var(--text-secondary); margin-bottom: 12px;">${env.os} • ${env.runtime}</p>
        <table class="data-table" style="font-size: 11px;">
          <tbody>
            <tr><td class="text-tertiary">Compute Limit</td><td class="table-mono">${env.cpu}, ${env.memory}</td></tr>
            <tr><td class="text-tertiary">GPU Acceleration</td><td class="table-mono">${env.gpu}</td></tr>
            <tr><td class="text-tertiary">Timeout SLA</td><td class="table-mono">${env.timeout}</td></tr>
            <tr><td class="text-tertiary">Network Policy</td><td><span class="badge badge-warn">${env.network}</span></td></tr>
          </tbody>
        </table>
      </div>
    `).join('');
  }

  function renderAnalytics() {
    const tbodyCal = document.getElementById('tbody-difficulty-calibration');
    if (tbodyCal && DATA.analytics && DATA.analytics.difficulty_calibration) {
      tbodyCal.innerHTML = DATA.analytics.difficulty_calibration.map(c => `
        <tr>
          <td><strong>${c.task}</strong></td>
          <td class="table-mono">${c.estimated_mins} mins</td>
          <td class="table-mono">${c.actual_agent_mins} mins</td>
          <td class="table-mono">${c.actual_human_mins} mins</td>
        </tr>
      `).join('');
    }

    const failureList = document.getElementById('analytics-failure-modes-list');
    if (failureList && DATA.analytics && DATA.analytics.failure_modes) {
      failureList.innerHTML = DATA.analytics.failure_modes.map(mode => `
        <div class="breakdown-row">
          <div class="breakdown-meta">
            <span class="breakdown-name">${mode.name}</span>
            <span class="breakdown-val">${mode.count} (${mode.pct})</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill ${mode.count > 10 ? 'danger' : 'warning'}" style="width: ${mode.pct};"></div>
          </div>
        </div>
      `).join('');
    }
  }

  function renderSystemHealth() {
    const tbody = document.getElementById('tbody-system-health');
    if (!tbody || !DATA.system_health) return;

    tbody.innerHTML = DATA.system_health.map(s => `
      <tr>
        <td><strong>${s.service}</strong></td>
        <td><span class="badge badge-pass">${s.status}</span></td>
        <td class="table-mono">${s.latency}</td>
        <td class="table-mono">${s.uptime}</td>
        <td style="color: var(--text-secondary);">${s.details}</td>
      </tr>
    `).join('');
  }

  function renderSettings() {
    const tbodyKeys = document.getElementById('tbody-settings-apikeys');
    if (!tbodyKeys || !DATA.settings || !DATA.settings.api_keys) return;

    tbodyKeys.innerHTML = DATA.settings.api_keys.map(k => `
      <tr>
        <td><strong>${k.name}</strong></td>
        <td class="table-mono">${k.prefix}</td>
        <td>${k.last_used}</td>
      </tr>
    `).join('');
  }

  // =========================================================================
  // Create Benchmark Wizard (Section 12)
  // =========================================================================

  function handleWizardNext() {
    if (wizardStep < 8) {
      wizardStep++;
      updateWizardUI();
    }
  }

  function updateWizardUI() {
    document.querySelectorAll('.wizard-step').forEach(stepEl => {
      const stepNum = parseInt(stepEl.getAttribute('data-step'), 10);
      stepEl.classList.toggle('active', stepNum === wizardStep);
      stepEl.classList.toggle('completed', stepNum < wizardStep);
    });

    for (let i = 1; i <= 8; i++) {
      const contentEl = document.getElementById(`step-content-${i}`);
      if (contentEl) {
        contentEl.style.display = (i === wizardStep) ? 'block' : 'none';
      }
    }

    const nextBtn = document.getElementById('btn-wizard-next');
    if (nextBtn) {
      nextBtn.style.display = (wizardStep === 8) ? 'none' : 'inline-flex';
    }
  }

  function handlePublishBenchmark() {
    showToast('Benchmark published successfully! Cryptographic manifest signed.', 'success');
    wizardStep = 1;
    updateWizardUI();
    navigateTo('benchmarks');
  }

  // =========================================================================
  // Command Palette (⌘K)
  // =========================================================================

  function initCommandPalette() {
    const modal = document.getElementById('command-palette-modal');
    const input = document.getElementById('palette-search-input');
    const trigger = document.getElementById('btn-open-palette');

    if (!modal || !input) return;

    function openPalette() {
      modal.classList.add('open');
      input.value = '';
      renderPaletteResults('');
      setTimeout(() => input.focus(), 50);
    }

    function closePalette() {
      modal.classList.remove('open');
    }

    if (trigger) trigger.addEventListener('click', openPalette);

    modal.addEventListener('click', (e) => {
      if (e.target === modal) closePalette();
    });

    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (modal.classList.contains('open')) closePalette();
        else openPalette();
      } else if (e.key === 'Escape') {
        closePalette();
        closeEvalModal();
      }
    });

    input.addEventListener('input', () => {
      renderPaletteResults(input.value.trim().toLowerCase());
    });
  }

  function renderPaletteResults(query) {
    const list = document.getElementById('palette-results-list');
    if (!list) return;

    const quickActions = [
      { name: 'Run Evaluation', view: 'runs', action: () => openEvalModal() },
      { name: 'Create Benchmark', view: 'create-benchmark', action: () => navigateTo('create-benchmark') },
      { name: 'View System Health', view: 'system-health', action: () => navigateTo('system-health') },
      { name: 'Audit Graders & Anti-Cheat', view: 'graders', action: () => navigateTo('graders') },
      { name: 'Compare Runs', view: 'compare', action: () => navigateTo('compare') }
    ];

    const benchmarks = (DATA.benchmarks || []).map(b => ({
      name: `${b.id} • ${b.name}`,
      view: 'benchmark-detail',
      id: b.id
    }));

    const runs = (DATA.runs || []).map(r => ({
      name: `Run ${r.id} (${r.status} - ${r.score} pts)`,
      view: 'run-detail',
      id: r.id
    }));

    const allItems = [
      ...quickActions.map(a => ({ ...a, group: 'Actions' })),
      ...benchmarks.map(b => ({ ...b, group: 'Benchmarks' })),
      ...runs.map(r => ({ ...r, group: 'Runs' }))
    ];

    const filtered = allItems.filter(item => item.name.toLowerCase().includes(query));

    if (filtered.length === 0) {
      list.innerHTML = `<div style="padding: 12px; text-align: center; color: var(--text-tertiary); font-size: 12px;">No matching actions, benchmarks, or runs.</div>`;
      return;
    }

    list.innerHTML = filtered.slice(0, 10).map((item, idx) => `
      <div class="palette-item ${idx === 0 ? 'selected' : ''}" onclick="window.ForgeEvalApp.selectPaletteItem('${item.view}', '${item.id || ''}')">
        <span>${item.name}</span>
        <span class="badge badge-neutral">${item.group}</span>
      </div>
    `).join('');
  }

  // =========================================================================
  // Evaluation Modal
  // =========================================================================

  function initModals() {
    const evalModal = document.getElementById('run-eval-modal');
    const closeBtn = document.getElementById('btn-close-eval-modal');
    const cancelBtn = document.getElementById('btn-cancel-eval-modal');
    const submitBtn = document.getElementById('btn-submit-eval-modal');

    if (closeBtn) closeBtn.addEventListener('click', closeEvalModal);
    if (cancelBtn) cancelBtn.addEventListener('click', closeEvalModal);

    if (submitBtn) {
      submitBtn.addEventListener('click', () => {
        closeEvalModal();
        showToast('Evaluation run submitted to runner queue...', 'info');
        setTimeout(() => {
          showToast('Run run_8f2a91 completed with score 92.0/100', 'success');
          navigateTo('run-detail', { id: 'run_8f2a91' });
        }, 1200);
      });
    }

    const btnNewRunOverview = document.getElementById('btn-overview-new-run');
    if (btnNewRunOverview) btnNewRunOverview.addEventListener('click', openEvalModal);

    const btnNewRunRuns = document.getElementById('btn-runs-trigger');
    if (btnNewRunRuns) btnNewRunRuns.addEventListener('click', openEvalModal);
  }

  function openEvalModal() {
    const modal = document.getElementById('run-eval-modal');
    if (modal) modal.classList.add('open');
  }

  function closeEvalModal() {
    const modal = document.getElementById('run-eval-modal');
    if (modal) modal.classList.remove('open');
  }

  // =========================================================================
  // State Gallery Switcher (Section 35, 36, 37 Demo)
  // =========================================================================

  function initStateSwitcher() {
    document.querySelectorAll('.state-switcher-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.state-switcher-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeState = btn.getAttribute('data-state');
        applyStateMode(activeState);
      });
    });
  }

  function applyStateMode(state) {
    const viewport = document.getElementById('views-viewport');
    if (!viewport) return;

    if (state === 'normal') {
      renderAllViews();
      showToast('Restored production state', 'info');
    } else if (state === 'skeleton') {
      // Show realistic skeleton loader
      const activePane = document.querySelector('.view-pane.active');
      if (activePane) {
        activePane.innerHTML = `
          <div class="view-header">
            <div class="skeleton skeleton-title" style="width: 200px;"></div>
          </div>
          <div class="kpi-grid">
            <div class="skeleton skeleton-card"></div>
            <div class="skeleton skeleton-card"></div>
            <div class="skeleton skeleton-card"></div>
            <div class="skeleton skeleton-card"></div>
          </div>
          <div class="panel-card" style="padding: 20px;">
            <div class="skeleton skeleton-text" style="width: 80%;"></div>
            <div class="skeleton skeleton-text" style="width: 60%;"></div>
            <div class="skeleton skeleton-text" style="width: 90%;"></div>
            <div class="skeleton skeleton-text" style="width: 40%;"></div>
          </div>
        `;
      }
      showToast('Simulating skeleton loading state...', 'info');
    } else if (state === 'empty') {
      const activePane = document.querySelector('.view-pane.active');
      if (activePane) {
        activePane.innerHTML = `
          <div class="state-container">
            <div class="state-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg>
            </div>
            <div class="state-title">No benchmarks created yet</div>
            <div class="state-desc">Create your first reproducible ML engineering benchmark to begin evaluating autonomous coding agents.</div>
            <button class="btn btn-primary btn-sm" onclick="window.ForgeEvalApp.navigateTo('create-benchmark')">+ Create Benchmark</button>
          </div>
        `;
      }
      showToast('Empty state displayed', 'info');
    } else if (state === 'error') {
      const activePane = document.querySelector('.view-pane.active');
      if (activePane) {
        activePane.innerHTML = `
          <div class="state-container" style="border-color: var(--danger-border); background: var(--danger-bg);">
            <div class="state-icon" style="background: rgba(239, 68, 68, 0.2); color: var(--danger-text);">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            </div>
            <div class="state-title" style="color: var(--danger-text);">Evaluation environment failed (Exit Code 137)</div>
            <div class="state-desc">Container was terminated by Docker host daemon: Memory limit exceeded (8192 MB limit breached during tensor allocation).</div>
            <div style="display: flex; gap: 8px;">
              <button class="btn btn-secondary btn-sm" onclick="window.ForgeEvalApp.openRun('run_8f2a91')">View Logs</button>
              <button class="btn btn-primary btn-sm" onclick="window.ForgeEvalApp.openEvalModal()">Retry with 16GB Limit</button>
            </div>
          </div>
        `;
      }
      showToast('Simulating Docker Exit 137 OOM Error State', 'danger');
    }
  }

  // =========================================================================
  // Toast Notification System
  // =========================================================================

  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'toast';

    let icon = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
    if (type === 'success') {
      icon = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981"><polyline points="20 6 9 17 4 12"/></svg>`;
    } else if (type === 'danger') {
      icon = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
    }

    toast.innerHTML = `${icon}<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.25s ease';
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // =========================================================================
  // Global App API Interface
  // =========================================================================

  window.ForgeEvalApp = {
    navigateTo,
    openBenchmark: (id) => navigateTo('benchmark-detail', { id }),
    openRun: (id) => navigateTo('run-detail', { id }),
    openAgent: (id) => navigateTo('agent-detail', { id }),
    openEvalModal,
    selectWorkspaceFile: (filePath) => {
      const tabTitle = document.getElementById('editor-active-tab');
      if (tabTitle) tabTitle.innerHTML = `<span>${filePath}</span>`;
      showToast(`Switched active buffer to ${filePath}`, 'info');
    },
    selectPaletteItem: (view, id) => {
      const modal = document.getElementById('command-palette-modal');
      if (modal) modal.classList.remove('open');
      navigateTo(view, { id });
    }
  };

})();
