package thompson

import (
	"encoding/json"
	"os"
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
	tmp := f.Path + ".tmp"
	if err := os.WriteFile(tmp, b, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, f.Path)
}

func (f *FileStore) Load() (*Snapshot, error) {
	b, err := os.ReadFile(f.Path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var s Snapshot
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}
	return &s, nil
}
