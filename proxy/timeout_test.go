package proxy

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"augur/trace"
)

// slowUpstream replies with chatResponse only after the given delay, unless the
// caller (the proxy) disconnects first — mirroring a slow reasoning model while
// still honoring cancellation.
func slowUpstream(delay time.Duration) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(delay):
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, chatResponse)
		case <-r.Context().Done():
		}
	}))
}

func postChat(t *testing.T, proxyURL string, hdr map[string]string) *http.Response {
	t.Helper()
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req, _ := http.NewRequest(http.MethodPost, proxyURL+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderScenarioID, "s")
	req.Header.Set(HeaderRunID, "r")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	return resp
}

// TestProxyTimeoutFires checks that a Server.Timeout shorter than the upstream's
// response time aborts the call promptly with a 502 instead of hanging.
func TestProxyTimeoutFires(t *testing.T) {
	up := slowUpstream(2 * time.Second)
	defer up.Close()
	u, _ := url.Parse(up.URL)

	var buf bytes.Buffer
	s := New(u, trace.NewWriter(&buf), up.Client())
	s.Timeout = 50 * time.Millisecond
	srv := httptest.NewServer(s)
	defer srv.Close()

	start := time.Now()
	resp := postChat(t, srv.URL, nil)
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	elapsed := time.Since(start)

	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want 502 on timeout", resp.StatusCode)
	}
	if elapsed > time.Second {
		t.Errorf("call took %v, want it to abort near the 50ms timeout", elapsed)
	}
}

// TestProxyTimeoutHeaderOverride checks a per-request X-Augur-Timeout narrows the
// deadline below a generous Server.Timeout.
func TestProxyTimeoutHeaderOverride(t *testing.T) {
	up := slowUpstream(2 * time.Second)
	defer up.Close()
	u, _ := url.Parse(up.URL)

	var buf bytes.Buffer
	s := New(u, trace.NewWriter(&buf), up.Client())
	s.Timeout = 10 * time.Minute // generous default
	srv := httptest.NewServer(s)
	defer srv.Close()

	start := time.Now()
	resp := postChat(t, srv.URL, map[string]string{HeaderTimeout: "50ms"})
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	elapsed := time.Since(start)

	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want 502 when the header timeout fires", resp.StatusCode)
	}
	if elapsed > time.Second {
		t.Errorf("call took %v, want the 50ms header timeout to win over Server.Timeout", elapsed)
	}
}

// TestProxyTimeoutHeaderStripped checks the X-Augur-Timeout header is Augur's own
// and never forwarded to the provider, and that a normal call still succeeds.
func TestProxyTimeoutHeaderStripped(t *testing.T) {
	up := newFakeUpstream()
	defer up.close()
	up.respBody = chatResponse

	var buf bytes.Buffer
	s := newTestProxy(t, up, &buf)
	srv := httptest.NewServer(s)
	defer srv.Close()

	resp := postChat(t, srv.URL, map[string]string{HeaderTimeout: "5m"})
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if len(up.gotReqs) != 1 {
		t.Fatalf("upstream got %d requests, want 1", len(up.gotReqs))
	}
	if got := up.gotReqs[0].header.Get(HeaderTimeout); got != "" {
		t.Errorf("X-Augur-Timeout leaked upstream: %q", got)
	}
}
