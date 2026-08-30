import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 20 },
    { duration: '1m', target: 50 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    'http_req_duration{scenario:health}': ['p(95)<100'],
    'http_req_duration{scenario:snapshots}': ['p(95)<200'],
    http_req_failed: ['rate<0.01'],
  },
};

const BASE = __ENV.BASE_URL || 'http://localhost:8080';
const TOKEN = __ENV.CONTROL_PLANE_TOKEN || '';

function headers() {
  const h = { 'Content-Type': 'application/json' };
  if (TOKEN) h['Authorization'] = `Bearer ${TOKEN}`;
  return h;
}

export default function () {
  let res = http.get(`${BASE}/health`, { tags: { scenario: 'health' } });
  check(res, { 'health 200': (r) => r.status === 200 });

  res = http.get(`${BASE}/metrics`, { tags: { scenario: 'metrics' } });
  check(res, { 'metrics 200': (r) => r.status === 200 && r.body.includes('traverse_snapshots_total_pulls') });

  res = http.get(`${BASE}/snapshots`, { headers: headers(), tags: { scenario: 'snapshots' } });
  check(res, { 'snapshots 200/401': (r) => r.status === 200 || r.status === 401 });

  sleep(1);
}
