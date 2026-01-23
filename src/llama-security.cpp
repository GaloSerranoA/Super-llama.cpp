#include "llama-security.h"
#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

namespace fs = std::filesystem;

// Global instances with thread-safe access
static llama_checkpoint_manager * g_checkpoint_manager = nullptr;
static llama_recovery_manager * g_recovery_manager = nullptr;
static std::mutex g_security_globals_mutex;

llama_checkpoint_manager * llama_get_checkpoint_manager() {
    std::lock_guard<std::mutex> lock(g_security_globals_mutex);
    return g_checkpoint_manager;
}
void llama_set_checkpoint_manager(llama_checkpoint_manager * m) {
    std::lock_guard<std::mutex> lock(g_security_globals_mutex);
    g_checkpoint_manager = m;
}

llama_recovery_manager * llama_get_recovery_manager() {
    std::lock_guard<std::mutex> lock(g_security_globals_mutex);
    return g_recovery_manager;
}
void llama_set_recovery_manager(llama_recovery_manager * m) {
    std::lock_guard<std::mutex> lock(g_security_globals_mutex);
    g_recovery_manager = m;
}

//
// Model Encryptor Implementation
//

llama_model_encryptor::llama_model_encryptor() = default;

llama_model_encryptor::llama_model_encryptor(const config & cfg) : cfg_(cfg) {}

llama_encryption_key llama_model_encryptor::generate_key() {
    llama_encryption_key key;
    key.algo = cfg_.algorithm;

    // Generate random key (32 bytes for AES-256)
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned int> dis(0, 255);

    key.key.resize(32);
    for (auto & byte : key.key) {
        byte = static_cast<uint8_t>(dis(gen));
    }

    // Generate random IV (12 bytes for GCM, 16 for CTR)
    int iv_size = (cfg_.algorithm == llama_encryption_algorithm::AES_256_GCM) ? 12 : 16;
    key.iv.resize(iv_size);
    for (auto & byte : key.iv) {
        byte = static_cast<uint8_t>(dis(gen));
    }

    return key;
}

llama_encryption_key llama_model_encryptor::derive_key(
        const std::string & password,
        const std::vector<uint8_t> & salt) {
    llama_encryption_key key;
    key.algo = cfg_.algorithm;

    // Simple PBKDF2-like derivation (placeholder - real impl would use proper PBKDF2)
    // In production, use OpenSSL's PKCS5_PBKDF2_HMAC or similar

    key.key.resize(32);
    key.iv.resize(16);

    // Simple derivation: hash password + salt repeatedly
    std::vector<uint8_t> derived;
    derived.reserve(password.size() + salt.size());
    derived.insert(derived.end(), password.begin(), password.end());
    derived.insert(derived.end(), salt.begin(), salt.end());

    // Simple hash-like operation (XOR-fold, not secure - placeholder only)
    for (int iter = 0; iter < cfg_.key_derivation_iterations; ++iter) {
        uint8_t hash = 0;
        for (size_t i = 0; i < derived.size(); ++i) {
            hash ^= derived[i];
            hash = (hash << 3) | (hash >> 5);  // Rotate
            derived[i] = hash;
        }
    }

    // Extract key and IV
    for (size_t i = 0; i < 32 && i < derived.size(); ++i) {
        key.key[i] = derived[i];
    }
    for (size_t i = 0; i < 16 && i + 32 < derived.size(); ++i) {
        key.iv[i] = derived[i + 32];
    }

    return key;
}

bool llama_model_encryptor::encrypt_model(
        const std::string & input_path,
        const std::string & output_path,
        const llama_encryption_key & key) {

    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        last_error_ = "Failed to open input file: " + input_path;
        return false;
    }

    // Read entire file
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(file_size);
    in.read(reinterpret_cast<char *>(data.data()), file_size);
    in.close();

    // Encrypt
    auto encrypted = encrypt_buffer(data.data(), data.size(), key);
    if (encrypted.empty()) {
        return false;
    }

    // Write header + encrypted data
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        last_error_ = "Failed to open output file: " + output_path;
        return false;
    }

    // Write magic number and version
    const char magic[] = "LLAMA_ENC";
    uint32_t version = 1;
    out.write(magic, 9);
    out.write(reinterpret_cast<const char *>(&version), sizeof(version));

    // Write algorithm
    uint32_t algo = static_cast<uint32_t>(key.algo);
    out.write(reinterpret_cast<const char *>(&algo), sizeof(algo));

    // Write IV
    uint32_t iv_size = static_cast<uint32_t>(key.iv.size());
    out.write(reinterpret_cast<const char *>(&iv_size), sizeof(iv_size));
    out.write(reinterpret_cast<const char *>(key.iv.data()), key.iv.size());

    // Write original size
    uint64_t orig_size = file_size;
    out.write(reinterpret_cast<const char *>(&orig_size), sizeof(orig_size));

    // Write checksum
    uint32_t checksum = calculate_checksum(data.data(), data.size());
    out.write(reinterpret_cast<const char *>(&checksum), sizeof(checksum));

    // Write encrypted data
    out.write(reinterpret_cast<const char *>(encrypted.data()), encrypted.size());
    out.close();

    LLAMA_LOG_INFO("%s: encrypted model %s -> %s (%zu bytes)\n",
        __func__, input_path.c_str(), output_path.c_str(), encrypted.size());

    return true;
}

bool llama_model_encryptor::decrypt_model(
        const std::string & input_path,
        const std::string & output_path,
        const llama_encryption_key & key) {

    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        last_error_ = "Failed to open input file: " + input_path;
        return false;
    }

    // Read and verify header
    char magic[10] = {0};
    in.read(magic, 9);
    if (std::string(magic) != "LLAMA_ENC") {
        last_error_ = "Invalid encrypted file format";
        return false;
    }

    uint32_t version;
    in.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != 1) {
        last_error_ = "Unsupported encryption version";
        return false;
    }

    uint32_t algo;
    in.read(reinterpret_cast<char *>(&algo), sizeof(algo));

    uint32_t iv_size;
    in.read(reinterpret_cast<char *>(&iv_size), sizeof(iv_size));
    std::vector<uint8_t> iv(iv_size);
    in.read(reinterpret_cast<char *>(iv.data()), iv_size);

    uint64_t orig_size;
    in.read(reinterpret_cast<char *>(&orig_size), sizeof(orig_size));

    uint32_t expected_checksum;
    in.read(reinterpret_cast<char *>(&expected_checksum), sizeof(expected_checksum));

    // Read encrypted data
    in.seekg(0, std::ios::end);
    size_t total_size = in.tellg();
    size_t header_size = 9 + sizeof(version) + sizeof(algo) + sizeof(iv_size) + iv_size +
                         sizeof(orig_size) + sizeof(expected_checksum);
    size_t encrypted_size = total_size - header_size;

    in.seekg(header_size, std::ios::beg);
    std::vector<uint8_t> encrypted(encrypted_size);
    in.read(reinterpret_cast<char *>(encrypted.data()), encrypted_size);
    in.close();

    // Create key with stored IV
    llama_encryption_key dec_key = key;
    dec_key.iv = iv;
    dec_key.algo = static_cast<llama_encryption_algorithm>(algo);

    // Decrypt
    auto decrypted = decrypt_buffer(encrypted.data(), encrypted.size(), dec_key);
    if (decrypted.empty()) {
        return false;
    }

    // Verify checksum
    if (cfg_.verify_integrity) {
        uint32_t actual_checksum = calculate_checksum(decrypted.data(), decrypted.size());
        if (actual_checksum != expected_checksum) {
            last_error_ = "Checksum verification failed - data may be corrupted";
            return false;
        }
    }

    // Write decrypted data
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        last_error_ = "Failed to open output file: " + output_path;
        return false;
    }
    out.write(reinterpret_cast<const char *>(decrypted.data()), decrypted.size());
    out.close();

    LLAMA_LOG_INFO("%s: decrypted model %s -> %s (%zu bytes)\n",
        __func__, input_path.c_str(), output_path.c_str(), decrypted.size());

    return true;
}

std::vector<uint8_t> llama_model_encryptor::encrypt_buffer(
        const uint8_t * data,
        size_t size,
        const llama_encryption_key & key) {

    if (data == nullptr || size == 0) {
        last_error_ = "Invalid input data";
        return {};
    }

    std::vector<uint8_t> result(size);
    std::memcpy(result.data(), data, size);

    // Apply encryption (simple XOR for placeholder - real impl would use AES-GCM)
    xor_encrypt(result.data(), result.size(), key.key.data(), key.key.size());

    return result;
}

std::vector<uint8_t> llama_model_encryptor::decrypt_buffer(
        const uint8_t * data,
        size_t size,
        const llama_encryption_key & key) {

    // XOR encryption is symmetric
    return encrypt_buffer(data, size, key);
}

bool llama_model_encryptor::verify_integrity(const std::string & path,
                                              const llama_encryption_key & key) {
    // Read file and verify checksum
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        last_error_ = "Failed to open file: " + path;
        return false;
    }

    // Read header to get expected checksum
    char magic[10] = {0};
    in.read(magic, 9);
    if (std::string(magic) != "LLAMA_ENC") {
        last_error_ = "Not an encrypted file";
        return false;
    }

    in.seekg(9 + 4 + 4, std::ios::beg);  // Skip magic, version, algo
    uint32_t iv_size;
    in.read(reinterpret_cast<char *>(&iv_size), sizeof(iv_size));
    in.seekg(iv_size, std::ios::cur);  // Skip IV

    uint64_t orig_size;
    in.read(reinterpret_cast<char *>(&orig_size), sizeof(orig_size));

    uint32_t expected_checksum;
    in.read(reinterpret_cast<char *>(&expected_checksum), sizeof(expected_checksum));

    in.close();

    // Note: Full verification would decrypt and check checksum
    // For now, just verify header is readable
    return true;
}

void llama_model_encryptor::xor_encrypt(uint8_t * data, size_t size,
                                         const uint8_t * key, size_t key_size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] ^= key[i % key_size];
    }
}

uint32_t llama_model_encryptor::calculate_checksum(const uint8_t * data, size_t size) {
    // Simple CRC32-like checksum (placeholder)
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    return ~crc;
}

//
// Checkpoint Manager Implementation
//

llama_checkpoint_manager::llama_checkpoint_manager() = default;

llama_checkpoint_manager::llama_checkpoint_manager(const config & cfg) : cfg_(cfg) {
    // Create checkpoint directory
    if (!fs::exists(cfg_.checkpoint_dir)) {
        fs::create_directories(cfg_.checkpoint_dir);
    }
}

llama_checkpoint_manager::~llama_checkpoint_manager() {
    if (g_checkpoint_manager == this) g_checkpoint_manager = nullptr;
}

bool llama_checkpoint_manager::save_checkpoint(const llama_checkpoint & checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string id = checkpoint.checkpoint_id;
    if (id.empty()) {
        id = generate_checkpoint_id();
    }

    std::string path = get_checkpoint_path(id);

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        LLAMA_LOG_WARN("%s: failed to create checkpoint file: %s\n", __func__, path.c_str());
        return false;
    }

    // Write header
    const char magic[] = "LLAMA_CKP";
    uint32_t version = 1;
    file.write(magic, 9);
    file.write(reinterpret_cast<const char *>(&version), sizeof(version));

    // Write checkpoint ID
    uint32_t id_len = static_cast<uint32_t>(id.size());
    file.write(reinterpret_cast<const char *>(&id_len), sizeof(id_len));
    file.write(id.data(), id.size());

    // Write timestamp and position
    file.write(reinterpret_cast<const char *>(&checkpoint.timestamp_ms), sizeof(checkpoint.timestamp_ms));
    file.write(reinterpret_cast<const char *>(&checkpoint.token_position), sizeof(checkpoint.token_position));

    // Write model and request IDs
    auto write_string = [&file](const std::string & s) {
        uint32_t len = static_cast<uint32_t>(s.size());
        file.write(reinterpret_cast<const char *>(&len), sizeof(len));
        file.write(s.data(), s.size());
    };

    write_string(checkpoint.model_id);
    write_string(checkpoint.request_id);

    // Write KV cache
    uint64_t kv_size = checkpoint.kv_cache_data.size();
    file.write(reinterpret_cast<const char *>(&kv_size), sizeof(kv_size));
    if (kv_size > 0) {
        file.write(reinterpret_cast<const char *>(checkpoint.kv_cache_data.data()), kv_size);
    }

    // Write RNG state
    uint32_t rng_size = static_cast<uint32_t>(checkpoint.rng_state.size());
    file.write(reinterpret_cast<const char *>(&rng_size), sizeof(rng_size));
    if (rng_size > 0) {
        file.write(reinterpret_cast<const char *>(checkpoint.rng_state.data()), rng_size);
    }

    // Write metadata
    uint32_t meta_count = static_cast<uint32_t>(checkpoint.metadata.size());
    file.write(reinterpret_cast<const char *>(&meta_count), sizeof(meta_count));
    for (const auto & [k, v] : checkpoint.metadata) {
        write_string(k);
        write_string(v);
    }

    file.close();

    // Cache it
    llama_checkpoint cached = checkpoint;
    cached.checkpoint_id = id;
    checkpoint_cache_[id] = cached;

    while (checkpoint_cache_.size() > MAX_CACHE_SIZE) {
        checkpoint_cache_.erase(checkpoint_cache_.begin());
    }

    LLAMA_LOG_INFO("%s: saved checkpoint %s (token %ld)\n",
        __func__, id.c_str(), checkpoint.token_position);

    return true;
}

std::optional<llama_checkpoint> llama_checkpoint_manager::load_checkpoint(
        const std::string & checkpoint_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check cache first
    auto it = checkpoint_cache_.find(checkpoint_id);
    if (it != checkpoint_cache_.end()) {
        return it->second;
    }

    std::string path = get_checkpoint_path(checkpoint_id);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }

    // Read and verify header
    char magic[10] = {0};
    file.read(magic, 9);
    if (std::string(magic) != "LLAMA_CKP") {
        return std::nullopt;
    }

    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != 1) {
        return std::nullopt;
    }

    llama_checkpoint checkpoint;

    // Read checkpoint ID
    uint32_t id_len;
    file.read(reinterpret_cast<char *>(&id_len), sizeof(id_len));
    checkpoint.checkpoint_id.resize(id_len);
    file.read(checkpoint.checkpoint_id.data(), id_len);

    // Read timestamp and position
    file.read(reinterpret_cast<char *>(&checkpoint.timestamp_ms), sizeof(checkpoint.timestamp_ms));
    file.read(reinterpret_cast<char *>(&checkpoint.token_position), sizeof(checkpoint.token_position));

    // Read strings
    auto read_string = [&file]() -> std::string {
        uint32_t len;
        file.read(reinterpret_cast<char *>(&len), sizeof(len));
        std::string s(len, '\0');
        file.read(s.data(), len);
        return s;
    };

    checkpoint.model_id = read_string();
    checkpoint.request_id = read_string();

    // Read KV cache
    uint64_t kv_size;
    file.read(reinterpret_cast<char *>(&kv_size), sizeof(kv_size));
    if (kv_size > 0) {
        checkpoint.kv_cache_data.resize(kv_size);
        file.read(reinterpret_cast<char *>(checkpoint.kv_cache_data.data()), kv_size);
    }
    checkpoint.kv_cache_size = kv_size;

    // Read RNG state
    uint32_t rng_size;
    file.read(reinterpret_cast<char *>(&rng_size), sizeof(rng_size));
    if (rng_size > 0) {
        checkpoint.rng_state.resize(rng_size);
        file.read(reinterpret_cast<char *>(checkpoint.rng_state.data()), rng_size);
    }

    // Read metadata
    uint32_t meta_count;
    file.read(reinterpret_cast<char *>(&meta_count), sizeof(meta_count));
    for (uint32_t i = 0; i < meta_count; ++i) {
        std::string k = read_string();
        std::string v = read_string();
        checkpoint.metadata[k] = v;
    }

    // Cache it
    checkpoint_cache_[checkpoint.checkpoint_id] = checkpoint;

    return checkpoint;
}

std::optional<llama_checkpoint> llama_checkpoint_manager::load_latest(
        const std::string & request_id) {
    auto checkpoints = list_checkpoints(request_id);
    if (checkpoints.empty()) {
        return std::nullopt;
    }

    // Sort by timestamp (descending)
    std::sort(checkpoints.begin(), checkpoints.end(),
        [](const llama_checkpoint & a, const llama_checkpoint & b) {
            return a.timestamp_ms > b.timestamp_ms;
        });

    return checkpoints.front();
}

std::vector<llama_checkpoint> llama_checkpoint_manager::list_checkpoints(
        const std::string & request_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<llama_checkpoint> result;

    for (const auto & entry : fs::directory_iterator(cfg_.checkpoint_dir)) {
        if (entry.path().extension() == ".ckp") {
            auto ckp = load_checkpoint(entry.path().stem().string());
            if (ckp && ckp->request_id == request_id) {
                result.push_back(*ckp);
            }
        }
    }

    return result;
}

bool llama_checkpoint_manager::delete_checkpoint(const std::string & checkpoint_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    checkpoint_cache_.erase(checkpoint_id);

    std::string path = get_checkpoint_path(checkpoint_id);
    return fs::remove(path);
}

void llama_checkpoint_manager::delete_request_checkpoints(const std::string & request_id) {
    auto checkpoints = list_checkpoints(request_id);
    for (const auto & ckp : checkpoints) {
        delete_checkpoint(ckp.checkpoint_id);
    }
}

void llama_checkpoint_manager::cleanup_old_checkpoints() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Group by request ID
    std::map<std::string, std::vector<std::pair<int64_t, std::string>>> request_checkpoints;

    for (const auto & entry : fs::directory_iterator(cfg_.checkpoint_dir)) {
        if (entry.path().extension() == ".ckp") {
            auto ckp = load_checkpoint(entry.path().stem().string());
            if (ckp) {
                request_checkpoints[ckp->request_id].emplace_back(
                    ckp->timestamp_ms, ckp->checkpoint_id);
            }
        }
    }

    // Keep only max_checkpoints per request
    for (auto & [req_id, checkpoints] : request_checkpoints) {
        if (static_cast<int>(checkpoints.size()) <= cfg_.max_checkpoints) {
            continue;
        }

        // Sort by timestamp (ascending)
        std::sort(checkpoints.begin(), checkpoints.end());

        // Delete oldest
        int to_delete = static_cast<int>(checkpoints.size()) - cfg_.max_checkpoints;
        for (int i = 0; i < to_delete; ++i) {
            delete_checkpoint(checkpoints[i].second);
        }
    }
}

bool llama_checkpoint_manager::should_checkpoint(int64_t current_tokens,
                                                   int64_t last_checkpoint_tokens) const {
    return cfg_.auto_checkpoint &&
           (current_tokens - last_checkpoint_tokens) >= cfg_.checkpoint_interval_tokens;
}

std::string llama_checkpoint_manager::get_checkpoint_path(const std::string & checkpoint_id) const {
    return cfg_.checkpoint_dir + "/" + checkpoint_id + ".ckp";
}

std::string llama_checkpoint_manager::generate_checkpoint_id() const {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    auto now = std::chrono::system_clock::now();
    int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    std::ostringstream oss;
    oss << "ckp_" << timestamp << "_" << std::hex << (dis(gen) & 0xFFFFFFFF);
    return oss.str();
}

//
// Recovery Manager Implementation
//

llama_recovery_manager::llama_recovery_manager() = default;

llama_recovery_manager::llama_recovery_manager(const config & cfg) : cfg_(cfg) {}

void llama_recovery_manager::report_failure(const std::string & request_id,
                                             llama_failure_type type,
                                             const std::string & error_message) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto & state = recovery_states_[request_id];
    state.request_id = request_id;
    state.failure_type = type;
    state.last_error = error_message;
    state.retry_count++;

    // Call failure callback
    if (cfg_.on_failure) {
        cfg_.on_failure(type, error_message);
    }

    // Save checkpoint if enabled
    if (cfg_.checkpoint_on_error && checkpoint_manager_) {
        // Note: Would need context to actually save KV cache
        LLAMA_LOG_INFO("%s: failure reported for request %s: %s\n",
            __func__, request_id.c_str(), error_message.c_str());
    }
}

bool llama_recovery_manager::attempt_recovery(const std::string & request_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = recovery_states_.find(request_id);
    if (it == recovery_states_.end()) {
        return false;
    }

    auto & state = it->second;

    if (!cfg_.auto_recovery) {
        return false;
    }

    if (state.retry_count > cfg_.max_retries) {
        LLAMA_LOG_WARN("%s: max retries exceeded for request %s\n",
            __func__, request_id.c_str());
        return false;
    }

    // Delay before retry
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.retry_delay_ms));

    // Attempt to load checkpoint
    if (checkpoint_manager_ && !state.checkpoint_id.empty()) {
        auto ckp = checkpoint_manager_->load_checkpoint(state.checkpoint_id);
        if (ckp) {
            LLAMA_LOG_INFO("%s: recovered from checkpoint %s at token %ld\n",
                __func__, state.checkpoint_id.c_str(), ckp->token_position);
            state.recovered = true;
            return true;
        }
    }

    // Try recovery without checkpoint
    state.recovered = true;
    return true;
}

std::optional<llama_recovery_manager::recovery_state>
llama_recovery_manager::get_recovery_state(const std::string & request_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = recovery_states_.find(request_id);
    if (it == recovery_states_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void llama_recovery_manager::clear_recovery_state(const std::string & request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    recovery_states_.erase(request_id);
}

bool llama_recovery_manager::should_retry(const std::string & request_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = recovery_states_.find(request_id);
    if (it == recovery_states_.end()) {
        return false;
    }

    return cfg_.auto_recovery && it->second.retry_count <= cfg_.max_retries;
}

int llama_recovery_manager::get_reduced_batch_size(int current_batch_size) const {
    if (!cfg_.reduce_batch_on_oom) {
        return current_batch_size;
    }
    return std::max(1, current_batch_size / 2);
}

//
// TLS Context Implementation
//

llama_tls_context::llama_tls_context() = default;

llama_tls_context::llama_tls_context(const llama_tls_config & cfg) : cfg_(cfg) {}

llama_tls_context::~llama_tls_context() = default;

bool llama_tls_context::initialize() {
    // Note: Real implementation would initialize OpenSSL/BoringSSL context
    // This is a placeholder

    if (!cfg_.cert_file.empty() && !fs::exists(cfg_.cert_file)) {
        error_ = "Certificate file not found: " + cfg_.cert_file;
        return false;
    }

    if (!cfg_.key_file.empty() && !fs::exists(cfg_.key_file)) {
        error_ = "Key file not found: " + cfg_.key_file;
        return false;
    }

    initialized_ = true;
    LLAMA_LOG_INFO("%s: TLS context initialized (TLS 1.%d minimum)\n",
        __func__, cfg_.min_tls_version);
    return true;
}

std::vector<uint8_t> llama_tls_context::encrypt(const uint8_t * data, size_t size) {
    if (!initialized_) {
        return {};
    }

    // Placeholder: just copy data
    // Real implementation would use TLS record layer encryption
    return std::vector<uint8_t>(data, data + size);
}

std::vector<uint8_t> llama_tls_context::decrypt(const uint8_t * data, size_t size) {
    if (!initialized_) {
        return {};
    }

    // Placeholder: just copy data
    return std::vector<uint8_t>(data, data + size);
}

bool llama_tls_context::verify_certificate(const std::string & hostname) {
    if (!initialized_) {
        return false;
    }

    // Placeholder: always succeed
    // Real implementation would verify certificate chain
    return true;
}

//
// API Key Manager Implementation
//

llama_api_key_manager::llama_api_key_manager() = default;

llama_api_key_manager::llama_api_key_manager(const config & cfg) : cfg_(cfg) {}

std::pair<std::string, llama_api_key> llama_api_key_manager::generate_key(
        const std::string & client_id,
        const std::vector<std::string> & scopes,
        int64_t expiry_days) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string raw_key = generate_random_key();
    std::string key_hash = hash_key(raw_key);

    llama_api_key key;
    key.key_id = "key_" + key_hash.substr(0, 16);
    key.key_hash = key_hash;
    key.client_id = client_id;
    key.created_at_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    key.scopes = scopes;
    key.active = true;

    if (expiry_days > 0) {
        key.expires_at_ms = key.created_at_ms + (expiry_days * 24 * 60 * 60 * 1000LL);
    } else if (cfg_.require_expiry) {
        key.expires_at_ms = key.created_at_ms + (cfg_.default_expiry_days * 24 * 60 * 60 * 1000LL);
    }

    keys_by_hash_[key_hash] = key;
    key_id_to_hash_[key.key_id] = key_hash;

    // Return the raw key (only time it's available) and the key info
    return {raw_key, key};
}

bool llama_api_key_manager::validate_key(const std::string & key) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key_hash = hash_key(key);
    auto it = keys_by_hash_.find(key_hash);
    if (it == keys_by_hash_.end()) {
        return false;
    }

    const auto & key_info = it->second;

    if (!key_info.active) {
        return false;
    }

    if (key_info.expires_at_ms > 0) {
        int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        if (now > key_info.expires_at_ms) {
            return false;
        }
    }

    return true;
}

std::optional<llama_api_key> llama_api_key_manager::get_key_info(const std::string & key) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key_hash = hash_key(key);
    auto it = keys_by_hash_.find(key_hash);
    if (it == keys_by_hash_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<llama_api_key> llama_api_key_manager::get_key_by_id(const std::string & key_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = key_id_to_hash_.find(key_id);
    if (it == key_id_to_hash_.end()) {
        return std::nullopt;
    }

    auto key_it = keys_by_hash_.find(it->second);
    if (key_it == keys_by_hash_.end()) {
        return std::nullopt;
    }
    return key_it->second;
}

bool llama_api_key_manager::revoke_key(const std::string & key_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = key_id_to_hash_.find(key_id);
    if (it == key_id_to_hash_.end()) {
        return false;
    }

    auto key_it = keys_by_hash_.find(it->second);
    if (key_it != keys_by_hash_.end()) {
        key_it->second.active = false;
    }

    return true;
}

std::vector<llama_api_key> llama_api_key_manager::list_client_keys(const std::string & client_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<llama_api_key> result;
    for (const auto & [hash, key] : keys_by_hash_) {
        if (key.client_id == client_id) {
            result.push_back(key);
        }
    }
    return result;
}

void llama_api_key_manager::cleanup_expired_keys() {
    std::lock_guard<std::mutex> lock(mutex_);

    int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    std::vector<std::string> to_remove;
    for (const auto & [hash, key] : keys_by_hash_) {
        if (key.expires_at_ms > 0 && now > key.expires_at_ms) {
            to_remove.push_back(hash);
        }
    }

    for (const auto & hash : to_remove) {
        auto it = keys_by_hash_.find(hash);
        if (it != keys_by_hash_.end()) {
            key_id_to_hash_.erase(it->second.key_id);
            keys_by_hash_.erase(it);
        }
    }
}

bool llama_api_key_manager::has_scope(const std::string & key, const std::string & scope) {
    auto info = get_key_info(key);
    if (!info) return false;

    return std::find(info->scopes.begin(), info->scopes.end(), scope) != info->scopes.end();
}

std::string llama_api_key_manager::hash_key(const std::string & key) const {
    // Simple hash (placeholder - real impl would use SHA-256)
    uint64_t hash = 0;
    for (char c : key) {
        hash = hash * 31 + c;
    }

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return oss.str();
}

std::string llama_api_key_manager::generate_random_key() const {
    static const char charset[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, sizeof(charset) - 2);

    std::string key;
    key.reserve(cfg_.key_length);
    for (int i = 0; i < cfg_.key_length; ++i) {
        key += charset[dis(gen)];
    }

    return "sk_live_" + key;
}
