package thompson

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// SnapshotStore persists policy snapshots durably.
type SnapshotStore interface {
	Save(snapshot Snapshot) error
	Load() (*Snapshot, error)
}

// MemoryStore is an in-memory SnapshotStore for tests and harness use.
type MemoryStore struct {
	data []byte
}

func NewMemoryStore() *MemoryStore { return &MemoryStore{} }

func (m *MemoryStore) Save(s Snapshot) error {
	b, err := json.Marshal(s)
	if err != nil {
		return err
	}
	m.data = b
	return nil
}

func (m *MemoryStore) Load() (*Snapshot, error) {
	if m.data == nil {
		return nil, nil
	}
	var s Snapshot
	if err := json.Unmarshal(m.data, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// FileStore persists snapshots atomically via write-then-rename.
type FileStore struct {
	Path string
}

func NewFileStore(path string) *FileStore { return &FileStore{Path: path} }

func (f *FileStore) Save(s Snapshot) error {
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	if len(b) > 10*1024*1024 {
		return fmt.Errorf("thompson: snapshot too large (%d bytes)", len(b))
	}
	tmp := f.Path + ".tmp"
	// Use non-cached write with fsync for durability
	file, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	if _, err := file.Write(b); err != nil {
		file.Close()
		os.Remove(tmp)
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		os.Remove(tmp)
		return err
	}
	file.Close()
	if err := os.Rename(tmp, f.Path); err != nil {
		os.Remove(tmp)
		return err
	}
	// Best-effort fsync parent directory
	if dir, err := os.Open(filepath.Dir(f.Path)); err == nil {
		_ = dir.Sync()
		dir.Close()
	}
	return nil
}

func (f *FileStore) Load() (*Snapshot, error) {
	b, err := os.ReadFile(f.Path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	if len(b) > 10*1024*1024 {
		return nil, fmt.Errorf("thompson: snapshot too large (%d bytes)", len(b))
	}
	var s Snapshot
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}
	return &s, nil
}
