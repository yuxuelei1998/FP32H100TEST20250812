#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cctype>
#include <cmath>
#include <map>
#include <cstdint>

// 操作码枚举
enum Opcode {
    ADD, SUB, MUL, FMA, FMS, FNMA, FNMS,
    CMPEQ, CMPLT, CMPLE, CMPGT,
    CMPLTNUM, CMPLENUM, CMPGTNUM, UNORDERED
};

// 舍入模式枚举
enum RoundMode {
    RND_ZERO, RND_MINUS_INF, RND_PLUS_INF, RND_NEAREST
};

// 测试用例结构
struct TestCase {
    Opcode opcode;
    RoundMode roundMode;
    uint32_t operandA;
    uint32_t operandB;
    uint32_t operandC;
};

// 结果结构
struct Result {
    uint32_t result;
};

// 字符串到操作码映射
std::map<std::string, Opcode> opcodeMap = {
    {"ADD", ADD}, {"SUB", SUB}, {"MUL", MUL}, {"FMA", FMA}, {"FMS", FMS},
    {"FNMA", FNMA}, {"FNMS", FNMS}, {"CMPEQ", CMPEQ}, {"CMPLT", CMPLT},
    {"CMPLE", CMPLE}, {"CMPGT", CMPGT}, {"CMPLTNUM", CMPLTNUM},
    {"CMPLENUM", CMPLENUM}, {"CMPGTNUM", CMPGTNUM}, {"UNORDERED", UNORDERED}
};

// 字符串到舍入模式映射
std::map<std::string, RoundMode> roundModeMap = {
    {"RND_ZERO", RND_ZERO}, {"RND_MINUS_INF", RND_MINUS_INF},
    {"RND_PLUS_INF", RND_PLUS_INF}, {"RND_NEAREST", RND_NEAREST}
};

// CUDA内核：执行测试用例（仅原始计算）
__global__ void executeTests(const TestCase* testCases, Result* results, int numTests) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numTests) return;
    
    TestCase tc = testCases[idx];
    float a = __uint_as_float(tc.operandA);
    float b = __uint_as_float(tc.operandB);
    float c = __uint_as_float(tc.operandC);
    float res = 0.0f;
    
    // 直接硬件计算（无异常检测）
    switch (tc.opcode) {
        case ADD:
            if (tc.roundMode == RND_ZERO) res = __fadd_rz(a, c);
            else if (tc.roundMode == RND_MINUS_INF) res = __fadd_rd(a, c);
            else if (tc.roundMode == RND_PLUS_INF) res = __fadd_ru(a, c);
            else res = __fadd_rn(a, c);
            break;
        case SUB:
            if (tc.roundMode == RND_ZERO) res = __fsub_rz(a, c);
            else if (tc.roundMode == RND_MINUS_INF) res = __fsub_rd(a, c);
            else if (tc.roundMode == RND_PLUS_INF) res = __fsub_ru(a, c);
            else res = __fsub_rn(a, c);
            break;
        case MUL:
            if (tc.roundMode == RND_ZERO) res = __fmul_rz(a, b);
            else if (tc.roundMode == RND_MINUS_INF) res = __fmul_rd(a, b);
            else if (tc.roundMode == RND_PLUS_INF) res = __fmul_ru(a, b);
            else res = __fmul_rn(a, b);
            break;
        case FMA:
            if (tc.roundMode == RND_ZERO) res = __fmaf_rz(a, b, c);
            else if (tc.roundMode == RND_MINUS_INF) res = __fmaf_rd(a, b, c);
            else if (tc.roundMode == RND_PLUS_INF) res = __fmaf_ru(a, b, c);
            else res = __fmaf_rn(a, b, c);
            break;
        case FMS:
            if (tc.roundMode == RND_ZERO) res = __fmaf_rz(a, b, -c);
            else if (tc.roundMode == RND_MINUS_INF) res = __fmaf_rd(a, b, -c);
            else if (tc.roundMode == RND_PLUS_INF) res = __fmaf_ru(a, b, -c);
            else res = __fmaf_rn(a, b, -c);
            break;
        case FNMA:
            if (tc.roundMode == RND_ZERO) res = __fmaf_rz(-a, b, c);
            else if (tc.roundMode == RND_MINUS_INF) res = __fmaf_rd(-a, b, c);
            else if (tc.roundMode == RND_PLUS_INF) res = __fmaf_ru(-a, b, c);
            else res = __fmaf_rn(-a, b, c);
            break;
        case FNMS:
            if (tc.roundMode == RND_ZERO) res = __fmaf_rz(-a, b, -c);
            else if (tc.roundMode == RND_MINUS_INF) res = __fmaf_rd(-a, b, -c);
            else if (tc.roundMode == RND_PLUS_INF) res = __fmaf_ru(-a, b, -c);
            else res = __fmaf_rn(-a, b, -c);
            break;
        case CMPEQ:
            res = (a == c) ? __int_as_float(0xFFFFFFFF) : 0.0f;
            break;
        case CMPLT:
            res = (a < c) ? __int_as_float(0xFFFFFFFF) : 0.0f;
            break;
        case CMPLE:
            res = (a <= c) ? __int_as_float(0xFFFFFFFF) : 0.0f;
            break;
        case CMPGT:
            res = (a > c) ? __int_as_float(0xFFFFFFFF) : 0.0f;
            break;
        case CMPLTNUM:
            if (isnan(a)) {
                res = a;
            } else if (isnan(c)) {
                res = c;
            } else {
                res = (a < c) ? a : c;
            }
            break;
        case CMPLENUM:
            if (isnan(a)) {
                res = a;
            } else if (isnan(c)) {
                res = c;
            } else {
                res = (a <= c) ? a : c;
            }
            break;
        case CMPGTNUM:
            if (isnan(a)) {
                res = a;
            } else if (isnan(c)) {
                res = c;
            } else {
                res = (a > c) ? a : c;
            }
            break;
        case UNORDERED:
            res = (isnan(a) || isnan(c)) ? __int_as_float(0xFFFFFFFF) : 0.0f;
            break;
    }
    
    // 存储原始结果
    results[idx].result = __float_as_uint(res);
}

// 解析十六进制字符串
uint32_t parseHex(const std::string& hexStr) {
    return std::stoul(hexStr, nullptr, 16);
}

// 读取输入文件
std::vector<TestCase> readInputFile(const std::string& filename) {
    std::vector<TestCase> testCases;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(iss, token, ',')) {
            token.erase(0, token.find_first_not_of(' '));
            token.erase(token.find_last_not_of(' ') + 1);
            tokens.push_back(token);
        }
        
        if (tokens.size() == 5) {
            TestCase tc;
            tc.opcode = opcodeMap[tokens[0]];
            tc.roundMode = roundModeMap[tokens[1]];
            tc.operandA = parseHex(tokens[2]);
            tc.operandB = parseHex(tokens[3]);
            tc.operandC = parseHex(tokens[4]);
            testCases.push_back(tc);
        }
    }
    
    return testCases;
}

// 写输出文件（简化版）
void writeOutputFile(const std::string& filename, 
                    const std::vector<TestCase>& testCases,
                    const std::vector<Result>& results) {
    std::ofstream file(filename);
    file << "Opcode, Rnd, Operand A, Operand B, Operand C, Result\n";
    
    // 反向映射用于输出
    std::map<Opcode, std::string> opcodeStr;
    for (const auto& p : opcodeMap) opcodeStr[p.second] = p.first;
    
    std::map<RoundMode, std::string> roundModeStr;
    for (const auto& p : roundModeMap) roundModeStr[p.second] = p.first;
    
    for (size_t i = 0; i < testCases.size(); ++i) {
        const TestCase& tc = testCases[i];
        const Result& res = results[i];
        
        file << opcodeStr[tc.opcode] << ", "
             << roundModeStr[tc.roundMode] << ", "
             << "0x" << std::hex << std::setw(8) << std::setfill('0') << tc.operandA << ", "
             << "0x" << std::hex << std::setw(8) << std::setfill('0') << tc.operandB << ", "
             << "0x" << std::hex << std::setw(8) << std::setfill('0') << tc.operandC << ", "
             << "0x" << std::hex << std::setw(8) << std::setfill('0') << res.result << "\n";
    }
}

int main() {
    // 读取输入文件
    std::vector<TestCase> testCases = readInputFile("input.txt");
    int numTests = testCases.size();
    
    // 分配设备内存
    TestCase* d_testCases;
    Result* d_results;
    cudaMalloc(&d_testCases, numTests * sizeof(TestCase));
    cudaMalloc(&d_results, numTests * sizeof(Result));
    
    // 复制数据到设备
    cudaMemcpy(d_testCases, testCases.data(), numTests * sizeof(TestCase), cudaMemcpyHostToDevice);
    
    // 启动内核
    int blockSize = 512;
    int gridSize = (numTests + blockSize - 1) / blockSize;
    executeTests<<<gridSize, blockSize>>>(d_testCases, d_results, numTests);
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    std::vector<Result> results(numTests);
    cudaMemcpy(results.data(), d_results, numTests * sizeof(Result), cudaMemcpyDeviceToHost);
    
    // 写输出文件
    writeOutputFile("h100test_output.txt", testCases, results);
    
    // 清理
    cudaFree(d_testCases);
    cudaFree(d_results);
    
    std::cout << "H100 FP32 测试完成，结果已写入 h100test_output.txt" << std::endl;
    return 0;
}