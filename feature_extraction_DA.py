import re
import mysql.connector
from openai import OpenAI
from datetime import datetime
import hashlib
import time
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'feature_extraction_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL')
)

# MySQL配置
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

def get_hash(attribute_name, attribute_value):
    """生成属性名和属性值的组合哈希"""
    combined = f"{attribute_name}:{attribute_value}"
    return hashlib.sha256(combined.encode()).hexdigest()

def get_cached_feature(attribute_name, attribute_value):
    """从数据库中获取缓存的特征词"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        hash_value = get_hash(attribute_name, attribute_value)
        cursor.execute('''
            SELECT feature_word FROM DA_cache 
            WHERE hash_value = %s
        ''', (hash_value,))
        
        result = cursor.fetchone()
        
        return result['feature_word'] if result else None
    except Exception as e:
        logger.error(f"数据库查询错误: {str(e)}")
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def save_feature(attribute_name, attribute_value, feature_word):
    """保存特征词到数据库"""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    hash_value = get_hash(attribute_name, attribute_value)
    created_at = datetime.now()
    
    try:
        cursor.execute('''
            INSERT INTO DA_cache 
            (attribute_name, attribute_value, feature_word, hash_value, created_at)
            VALUES (%s, %s, %s, %s, %s)
        ''', (attribute_name, attribute_value, feature_word, hash_value, created_at))
        
        conn.commit()
    except mysql.connector.IntegrityError:
        # 如果哈希值已存在，忽略错误
        pass
    finally:
        cursor.close()
        conn.close()

def parse_entity(entity_str):
    """解析单个实体的属性（保持原有逻辑）"""
    entity_str = entity_str + ' COL'
    attributes = [f"COL {attr_str}" for attr_str 
                 in re.findall(r"(?<=COL ).*?(?= COL)", entity_str)]
    
    attr_dict = {}
    for attr in attributes:
        parts = attr.split(' VAL ')
        if len(parts) == 2:
            key = parts[0].replace('COL ', '').strip()
            value = parts[1].strip()
            attr_dict[key] = value
            
    return attr_dict

def generate_llm_prompt(attr_name, attr_value):
    """生成LLM提示词"""
    prompt = f"""作为资深学术文献专家，从论文信息中提取最具代表性的特征词。

要求:
1. 分析 {attr_name}: "{attr_value}"，提取1-2个最能体现该论文特色的关键词
2. 优先选择体现论文研究领域、核心技术、应用场景的专业词汇
3. 可以基于你的计算机科学领域知识，提取原文中未出现但更准确的特征词
4. 如果找不到明显特征，则返回基础研究方向(如Database、Data Mining、Information System等)

示例：
输入: title="Efficient Query Processing in Geographic Information Systems"
输出: "Spatial Database" # 体现核心技术领域

输入: title="Real-time Data Analytics in Cloud Computing"
输出: "Stream Processing" # 体现应用场景和技术平台

输入: venue="VLDB"
输出: "Database Systems"

输入: venue="SIGMOD"
输出: "Database Systems"

仅返回特征词，无需解释。

请分析并提取:"""
    
    return prompt

def get_llm_response(prompt):
    """调用DeepSeek API获取响应"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a computer science expert who extracts key features from academic papers and venues."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API调用错误: {str(e)}")
        return None

def clean_feature_word(feature_word):
    """清理特征词，去除双引号和多余的空格，保留1-2个词"""
    if feature_word is None:
        return None
    # 去除双引号和首尾空格
    cleaned = feature_word.strip('" \'')
    # 分割成词列表
    words = cleaned.split()
    # 只保留前两个词（如果有的话）
    if len(words) > 2:
        words = words[:2]
    # 用空格连接词
    cleaned = ' '.join(words)
    return cleaned

def extract_and_cache_feature(attribute_name, attribute_value):
    """提取特征并缓存"""
    # 首先尝试从缓存获取
    cached_feature = get_cached_feature(attribute_name, attribute_value)
    if cached_feature:
        return clean_feature_word(cached_feature), True  # True表示是从缓存获取的
    
    # 如果缓存中没有，调用LLM
    prompt = generate_llm_prompt(attribute_name, attribute_value)
    feature = get_llm_response(prompt)
    
    if feature:
        # 清理特征词
        cleaned_feature = clean_feature_word(feature)
        # 保存到数据库
        save_feature(attribute_name, attribute_value, cleaned_feature)
        time.sleep(3)  # 增加到3秒的延时
    
    return cleaned_feature, False  # False表示是新生成的

def process_line(line):
    """处理一行数据，提取两个实体的属性"""
    # 分割实体对
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None
    
    entity1 = parts[0]
    entity2 = parts[1]
    
    # 解析两个实体
    entity1_attrs = parse_entity(entity1)
    entity2_attrs = parse_entity(entity2)
    
    return entity1_attrs, entity2_attrs

def validate_entity(entity_attrs):
    """验证实体属性是否包含必要字段"""
    required_fields = ['title', 'venue']
    for field in required_fields:
        if field not in entity_attrs:
            return False
    return True

def extract_features(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            try:
                entity1_attrs, entity2_attrs = process_line(line)
                
                # 仅对第一条记录打印完整信息
                if i == 0:
                    print("Debug - First record attributes:")
                    print("Entity 1:", entity1_attrs)
                    print("Entity 2:", entity2_attrs)
                    print("-" * 50)
                
                # 验证实体属性
                if not validate_entity(entity1_attrs) or not validate_entity(entity2_attrs):
                    print(f"Warning: Line {i+1} missing required fields")
                    continue
                
                print(f"\n=== 实体对 {i+1} ===")
                
                # 处理实体1
                print("实体1:")
                for attr_name in ['title', 'venue']:
                    if attr_name in entity1_attrs:
                        attr_value = entity1_attrs[attr_name]
                        feature, is_cached = extract_and_cache_feature(attr_name, attr_value)
                        print(f"{attr_name}: {attr_value}")
                        print(f"提取的特征词: {feature} {'(已缓存)' if is_cached else '(新生成)'}")
                print()
                
                # 处理实体2
                print("实体2:")
                for attr_name in ['title', 'venue']:
                    if attr_name in entity2_attrs:
                        attr_value = entity2_attrs[attr_name]
                        feature, is_cached = extract_and_cache_feature(attr_name, attr_value)
                        print(f"{attr_name}: {attr_value}")
                        print(f"提取的特征词: {feature} {'(已缓存)' if is_cached else '(新生成)'}")
                print()
                
                print("-" * 50)
                
            except Exception as e:
                print(f"Error processing line {i+1}: {str(e)}")

def process_dataset(filename):
    """处理单个数据集文件"""
    logger.info(f"\n开始处理数据集: {filename}")
    start_time = time.time()
    
    try:
        extract_features(filename)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"完成处理数据集: {filename}")
        logger.info(f"处理耗时: {duration:.2f} 秒")
    except Exception as e:
        logger.error(f"处理数据集 {filename} 时发生错误: {str(e)}")

if __name__ == "__main__":
    logger.info("=== 开始特征提取任务 ===")
    
    # 检查环境变量
    required_env_vars = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"缺少必要的环境变量: {', '.join(missing_vars)}")
        exit(1)
    
    # 测试数据库连接
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        conn.close()
        logger.info("数据库连接测试成功")
    except Exception as e:
        logger.error(f"数据库连接失败: {str(e)}")
        exit(1)
    
    # 定义要处理的数据集文件
    datasets = [
        'train.txt',
        'test.txt',
        'valid.txt'
    ]
    
    # 批量处理所有数据集
    for dataset in datasets:
        try:
            process_dataset(dataset)
        except FileNotFoundError:
            logger.warning(f"警告: 文件 {dataset} 不存在，已跳过")
        except Exception as e:
            logger.error(f"处理文件 {dataset} 时发生错误: {str(e)}")
    
    logger.info("=== 特征提取任务完成 ===")