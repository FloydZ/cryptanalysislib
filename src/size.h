constexpr int get_string_size(const char * input,int i = 0) {
    return input[i] == '\0' ? i : get_string_size(input,i+1);
}

template<typename T>
constexpr size_t get_size(T input) {
    return sizeof(T);
}

template<>
constexpr size_t get_size(const char * input) {
    return get_string_size(input);
}

template<>
constexpr size_t get_size(char * input) {
    return get_string_size(input);
}

template<>
size_t get_size(std::string input) {
    return input.size();
}
