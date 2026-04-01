import { useEffect, useState } from 'react';
import axios from 'axios';

const API_BASE = '/api';

export default function App() {
  const [inputText, setInputText] = useState('');
  const [inputPath, setInputPath] = useState('');
  const [mock, setMock] = useState(true);
  const [workers, setWorkers] = useState(1);
  const [agent, setAgent] = useState(false);
  const [steering, setSteering] = useState('auto');
  const [threshold, setThreshold] = useState(0.7);
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [logs, setLogs] = useState('');
  const [loading, setLoading] = useState(false);
  const [submitStatus, setSubmitStatus] = useState('');

  async function fetchRuns() {
    try {
      const res = await axios.get(`${API_BASE}/runs`);
      setRuns(res.data);
    } catch (err) {
      console.error('fetchRuns', err);
    }
  }

  async function fetchRunDetails(runId) {
    try {
      const res = await axios.get(`${API_BASE}/runs/${runId}`);
      setSelectedRun(res.data);
    } catch (err) {
      console.error('fetchRunDetails', err);
      setSelectedRun(null);
    }
  }

  async function fetchRunLogs(runId) {
    try {
      const res = await axios.get(`${API_BASE}/runs/${runId}/logs`);
      setLogs(res.data.logs);
    } catch (err) {
      setLogs('No logs available yet.');
    }
  }

  const submitRun = async (e) => {
    e.preventDefault();

    if (!inputText && !inputPath) {
      setSubmitStatus('Please provide input text or input path.');
      return;
    }

    setLoading(true);
    setSubmitStatus('Creating run...');

    try {
      const body = {
        input_text: inputText || undefined,
        input_path: inputPath || undefined,
        mock,
        workers,
        agent,
        steering,
        threshold,
      };

      const res = await axios.post(`${API_BASE}/runs`, body);
      setSubmitStatus(`Run queued: ${res.data.run_id}`);
      setInputText('');
      setInputPath('');
      fetchRuns();

      const interval = setInterval(async () => {
        await fetchRunDetails(res.data.run_id);
        const runResponse = await axios.get(`${API_BASE}/runs/${res.data.run_id}`);
        if (runResponse.data.status === 'succeeded' || runResponse.data.status === 'failed') {
          clearInterval(interval);
          setSubmitStatus(`Run ${runResponse.data.run_id} ${runResponse.data.status}`);
          await fetchRunLogs(runResponse.data.run_id);
          fetchRuns();
        }
      }, 2000);
    } catch (err) {
      console.error('submitRun', err);
      setSubmitStatus('Failed to start run.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <h1>synthDataLab Production Dashboard</h1>
        <p>Monitor and launch synthetic dataset generation runs with health checks, logs, and pipeline controls.</p>
      </header>

      <div className="stats-grid">
        <div className="stat-card">
          <h3>Total runs</h3>
          <strong>{runs.length}</strong>
        </div>
        <div className="stat-card">
          <h3>Running</h3>
          <strong>{runs.filter((r) => r.status === 'running').length}</strong>
        </div>
        <div className="stat-card">
          <h3>Succeeded</h3>
          <strong>{runs.filter((r) => r.status === 'succeeded').length}</strong>
        </div>
        <div className="stat-card">
          <h3>Failed</h3>
          <strong>{runs.filter((r) => r.status === 'failed').length}</strong>
        </div>
      </div>

      <div className="dashboard-grid">
        <section className="panel panel-left">
          <h2>Create Run</h2>
          <form onSubmit={submitRun}>
            <label>
              Input text
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                rows={5}
                placeholder="Paste text sample to ingest"
              />
            </label>
            <label>
              Input path
              <input
                type="text"
                value={inputPath}
                onChange={(e) => setInputPath(e.target.value)}
                placeholder="/data/sample_inputs/sample_text.txt"
              />
            </label>
            <div className="row">
              <label>
                Mock LLM
                <select value={mock} onChange={(e) => setMock(e.target.value === 'true')}>
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              </label>
              <label>
                Agent
                <select value={agent} onChange={(e) => setAgent(e.target.value === 'true')}>
                  <option value="false">false</option>
                  <option value="true">true</option>
                </select>
              </label>
            </div>
            <div className="row">
              <label>
                Workers
                <input
                  type="number"
                  min="1"
                  max="16"
                  value={workers}
                  onChange={(e) => setWorkers(Number(e.target.value))}
                />
              </label>
              <label>
                Steering
                <select value={steering} onChange={(e) => setSteering(e.target.value)}>
                  <option value="auto">auto</option>
                  <option value="review-low">review-low</option>
                  <option value="review-all">review-all</option>
                </select>
              </label>
            </div>
            <label>
              Threshold {threshold}
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </label>
            <button type="submit" disabled={loading}>
              {loading ? 'Starting run...' : 'Start run'}
            </button>
            <p className="submit-status">{submitStatus}</p>
          </form>
        </section>

        <section className="panel panel-middle">
          <div className="panel-header">
            <h2>Run List</h2>
            <button onClick={fetchRuns}>Refresh</button>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Updated</th>
                  <th>Error</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.run_id} className={run.status === 'failed' ? 'failed' : run.status === 'running' ? 'running' : 'success'}>
                    <td>{run.run_id}</td>
                    <td>{run.status}</td>
                    <td>{new Date(run.updated_at).toLocaleString()}</td>
                    <td>{run.error || '-'}</td>
                    <td>
                      <button
                        className="small"
                        onClick={() => {
                          fetchRunDetails(run.run_id);
                          fetchRunLogs(run.run_id);
                        }}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel panel-right">
          <h2>Selected Run Details</h2>
          {selectedRun ? (
            <>
              <pre className="details-json">{JSON.stringify(selectedRun, null, 2)}</pre>
              <h3>Logs</h3>
              <pre className="logs">{logs || 'No logs yet'}</pre>
            </>
          ) : (
            <p>Select a run from the list to view details and logs.</p>
          )}
        </section>
      </div>
    </div>
  );
}
