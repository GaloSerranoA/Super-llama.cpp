#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

//
// Model Encryption - Encrypt model weights at rest and in transit
//

enum class llama_encryption_algorithm {
    AES_256_GCM,     // AES-256 in GCM mode (authenticated encryption)
    AES_256_CTR,     // AES-256 in CTR mode (stream cipher)
    CHACHA20_POLY1305, // ChaCha20-Poly1305 (modern, fast)
};

struct llama_encryption_key {
    std::vector<uint8_t> key;        // Encryption key (32 bytes for AES-256)
    std::vector<uint8_t> iv;         // Initialization vector
    llama_encryption_algorithm algo = llama_encryption_algorithm::AES_256_GCM;
};

class llama_model_encryptor {
public:
    struct config {
        llama_encryption_algorithm algorithm = llama_encryption_algorithm::AES_256_GCM;
        bool encrypt_weights = true;
        bool encrypt_metadata = true;
        bool verify_integrity = true;
        int key_derivation_iterations = 100000;  // PBKDF2 iterations
    };

    llama_model_encryptor();
    explicit llama_model_encryptor(const config & cfg);

    // Generate a new encryption key
    llama_encryption_key generate_key();

    // Derive key from password
    llama_encryption_key derive_key(const std::string & password, const std::vector<uint8_t> & salt);

    // Encrypt model file
    bool encrypt_model(const std::string & input_path,
                       const std::string & output_path,
                       const llama_encryption_key & key);

    // Decrypt model file
    bool decrypt_model(const std::string & input_path,
                       const std::string & output_path,
                       const llama_encryption_key & key);

    // Encrypt buffer in memory
    std::vector<uint8_t> encrypt_buffer(const uint8_t * data, size_t size,
                                         const llama_encryption_key & key);

    // Decrypt buffer in memory
    std::vector<uint8_t> decrypt_buffer(const uint8_t * data, size_t size,
                                         const llama_encryption_key & key);

    // Verify model integrity (check MAC/signature)
    bool verify_integrity(const std::string & path, const llama_encryption_key & key);

    // Get last error message
    std::string get_last_error() const { return last_error_; }

private:
    // Simple XOR-based encryption (placeholder - real impl would use proper crypto)
    void xor_encrypt(uint8_t * data, size_t size, const uint8_t * key, size_t key_size);

    // Calculate simple checksum (placeholder - real impl would use proper MAC)
    uint32_t calculate_checksum(const uint8_t * data, size_t size);

    config cfg_;
    std::string last_error_;
};

//
// Automatic Recovery with Checkpointing
//

struct llama_checkpoint {
    std::string checkpoint_id;
    int64_t timestamp_ms = 0;
    int64_t token_position = 0;
    std::string model_id;
    std::string request_id;

    // KV cache state
    std::vector<uint8_t> kv_cache_data;
    size_t kv_cache_size = 0;

    // Random state for reproducibility
    std::vector<uint8_t> rng_state;

    // Metadata
    std::map<std::string, std::string> metadata;
};

class llama_checkpoint_manager {
public:
    struct config {
        std::string checkpoint_dir = "./checkpoints";
        int max_checkpoints = 10;           // Per request
        int checkpoint_interval_tokens = 1000;
        bool auto_checkpoint = true;
        bool compress_checkpoints = true;
        bool encrypt_checkpoints = false;
    };

    llama_checkpoint_manager();
    explicit llama_checkpoint_manager(const config & cfg);
    ~llama_checkpoint_manager();

    // Save checkpoint
    bool save_checkpoint(const llama_checkpoint & checkpoint);

    // Load checkpoint
    std::optional<llama_checkpoint> load_checkpoint(const std::string & checkpoint_id);

    // Load latest checkpoint for request
    std::optional<llama_checkpoint> load_latest(const std::string & request_id);

    // List checkpoints for request
    std::vector<llama_checkpoint> list_checkpoints(const std::string & request_id);

    // Delete checkpoint
    bool delete_checkpoint(const std::string & checkpoint_id);

    // Delete all checkpoints for request
    void delete_request_checkpoints(const std::string & request_id);

    // Cleanup old checkpoints
    void cleanup_old_checkpoints();

    // Check if checkpoint should be created
    bool should_checkpoint(int64_t current_tokens, int64_t last_checkpoint_tokens) const;

    const config & get_config() const { return cfg_; }

private:
    std::string get_checkpoint_path(const std::string & checkpoint_id) const;
    std::string generate_checkpoint_id() const;

    config cfg_;
    mutable std::mutex mutex_;

    // In-memory checkpoint cache
    std::map<std::string, llama_checkpoint> checkpoint_cache_;
    static constexpr size_t MAX_CACHE_SIZE = 100;
};

//
// Automatic Recovery System
//

enum class llama_failure_type {
    OOM,              // Out of memory
    GPU_ERROR,        // GPU error/crash
    TIMEOUT,          // Request timeout
    INVALID_STATE,    // Invalid internal state
    UNKNOWN,          // Unknown error
};

class llama_recovery_manager {
public:
    struct config {
        bool auto_recovery = true;
        int max_retries = 3;
        int retry_delay_ms = 1000;
        bool checkpoint_on_error = true;
        bool reduce_batch_on_oom = true;
        std::function<void(llama_failure_type, const std::string &)> on_failure;
    };

    struct recovery_state {
        std::string request_id;
        llama_failure_type failure_type;
        int retry_count = 0;
        std::string last_error;
        std::string checkpoint_id;
        bool recovered = false;
    };

    llama_recovery_manager();
    explicit llama_recovery_manager(const config & cfg);

    // Report a failure
    void report_failure(const std::string & request_id,
                        llama_failure_type type,
                        const std::string & error_message);

    // Attempt recovery
    bool attempt_recovery(const std::string & request_id);

    // Get recovery state
    std::optional<recovery_state> get_recovery_state(const std::string & request_id) const;

    // Clear recovery state
    void clear_recovery_state(const std::string & request_id);

    // Check if request should be retried
    bool should_retry(const std::string & request_id) const;

    // Get suggested batch size after OOM
    int get_reduced_batch_size(int current_batch_size) const;

    const config & get_config() const { return cfg_; }

private:
    config cfg_;
    mutable std::mutex mutex_;
    std::map<std::string, recovery_state> recovery_states_;
    llama_checkpoint_manager * checkpoint_manager_ = nullptr;
};

//
// Secure Communication (TLS wrapper)
//

struct llama_tls_config {
    std::string cert_file;
    std::string key_file;
    std::string ca_file;
    bool verify_peer = true;
    bool require_client_cert = false;
    std::string ciphers = "HIGH:!aNULL:!MD5";
    int min_tls_version = 12;  // TLS 1.2
};

class llama_tls_context {
public:
    llama_tls_context();
    explicit llama_tls_context(const llama_tls_config & cfg);
    ~llama_tls_context();

    // Initialize TLS context
    bool initialize();

    // Encrypt data for transmission
    std::vector<uint8_t> encrypt(const uint8_t * data, size_t size);

    // Decrypt received data
    std::vector<uint8_t> decrypt(const uint8_t * data, size_t size);

    // Verify certificate
    bool verify_certificate(const std::string & hostname);

    // Get error message
    std::string get_error() const { return error_; }

    bool is_initialized() const { return initialized_; }

private:
    llama_tls_config cfg_;
    bool initialized_ = false;
    std::string error_;
};

//
// API Key Management
//

struct llama_api_key {
    std::string key_id;
    std::string key_hash;      // Hashed key (never store plaintext)
    std::string client_id;
    int64_t created_at_ms = 0;
    int64_t expires_at_ms = 0; // 0 = never expires
    bool active = true;
    std::vector<std::string> scopes;  // Permissions
    std::map<std::string, std::string> metadata;
};

class llama_api_key_manager {
public:
    struct config {
        int key_length = 32;
        std::string hash_algorithm = "sha256";
        bool require_expiry = false;
        int64_t default_expiry_days = 365;
    };

    llama_api_key_manager();
    explicit llama_api_key_manager(const config & cfg);

    // Generate new API key
    std::pair<std::string, llama_api_key> generate_key(
        const std::string & client_id,
        const std::vector<std::string> & scopes = {},
        int64_t expiry_days = -1);

    // Validate API key
    bool validate_key(const std::string & key);

    // Get key info (by key)
    std::optional<llama_api_key> get_key_info(const std::string & key);

    // Get key info (by key_id)
    std::optional<llama_api_key> get_key_by_id(const std::string & key_id);

    // Revoke key
    bool revoke_key(const std::string & key_id);

    // List keys for client
    std::vector<llama_api_key> list_client_keys(const std::string & client_id);

    // Cleanup expired keys
    void cleanup_expired_keys();

    // Check if key has scope
    bool has_scope(const std::string & key, const std::string & scope);

private:
    std::string hash_key(const std::string & key) const;
    std::string generate_random_key() const;

    config cfg_;
    mutable std::mutex mutex_;
    std::map<std::string, llama_api_key> keys_by_hash_;  // hash -> key info
    std::map<std::string, std::string> key_id_to_hash_;  // key_id -> hash
};

// Global instances
llama_checkpoint_manager * llama_get_checkpoint_manager();
void llama_set_checkpoint_manager(llama_checkpoint_manager * manager);

llama_recovery_manager * llama_get_recovery_manager();
void llama_set_recovery_manager(llama_recovery_manager * manager);
