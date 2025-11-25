"""Training script for noise-conditioned predictors.

This script trains molecular property predictors that work with noisy inputs,
which can be used for classifier-based guidance during diffusion model sampling.

Usage:
    python train_predictor.py -d 0 -e 100 -w 0.0
"""

# Import all utilities from predictor_utils
from predictor_utils import *

if __name__ == '__main__':
    # ============================================================================
    # ARGUMENT PARSING
    # ============================================================================
    parser = argparse.ArgumentParser(
        description='Train noise-conditioned predictor for classifier-guided generation'
    )
    parser.add_argument(
        '-p', '--parallel',  # 可选参数
        action='store_true',
        help='whether to parallel validation on multi GPUs'
    )
    parser.add_argument(
        '-t', '--test_group',  # 可选参数
        type=int,
        # choices=['Serinales', 'Betaproteobacteria', 'FCB', 'VPC', 'BFSP', 'Eurotiomycetes', 'MA', 'Bacillales', 'Enterobacterales', 'Lactobacillales', 'ALs'],  # 可选项列表
        default=None,
        help='which task to test on in this experiment'
    )
    parser.add_argument(
        '-d', '--device',  # 可选参数
        type=int,
        default=3,
        help='Which GPU to use'
    )
    parser.add_argument(
        '-e', '--epoch',  # 可选参数
        type=int,
        default=100,
        help='How many epochs to train'
    )
    parser.add_argument(
        '-w', '--weight_decay',  # 可选参数
        type=float,
        default=0,
        help='weight decay lambda'
    )
    args = parser.parse_args()
    if args.parallel and args.test_group is None:
        print('\n Please specify test group when parallel validation is on')
        exit(1)

    # ============================================================================
    # CONFIGURATION & INITIALIZATION
    # ============================================================================
    genome_embedding_scale_factor = 1e14
    text_embedding_scale_factor = 1
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # num_clusters = 11  # 给 细菌 species 聚类的数量
    num_ensembles = 1  # 要集成几个 model 来做预测
    random_seeds = [42, 2024, 2025, 2077, 2012, 1973, 2002, 2001, 2020, 2019, 31, 13, 55, 11, 12, 58, 72, 2010, 2008, 2001, 1717, 1313, 99, 83, 29, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027]
    model_save_dir = current_directory / 'Checkpoints' / 'genome_text_learnable_emb' / 'guidance_regressor_pad_no_mask'
    if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n {str(model_save_dir)} created！")
    else:
        print(f"\n {str(model_save_dir)} exist.")

    # ensemble 的数量和 random_seeds 的数量必须相等
    if len(random_seeds) < num_ensembles:
        print(f'\n num of randome seeds: {len(random_seeds)} should be equal to or bigger than the num of ensembles: {num_ensembles}')
        exit(1)

    # ============================================================================
    # LOGGING SETUP
    # ============================================================================
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件Handler
    file_handler = logging.FileHandler(model_save_dir/f'log_guidance_regressor.log', mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 创建控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    # 添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Training started")

    # ============================================================================
    # DATA LOADING & PREPROCESSING
    # ============================================================================
    embeddings_folder_path = current_directory / 'DataPrepare' / 'Data' / 'Genome_embs'
    text_embeddings_folder_path = current_directory / 'DataPrepare' / 'Data' / 'Text_Description' / 'ATCC' / 'embeddings'
    text_embeddings_wo_genome_folder_path = current_directory / 'DataPrepare' / 'Data' / 'Text_Description' / 'wo_ATCC' / 'embeddings'

    embedded_genome_IDs, genome_ID_to_species_first_name_dict = get_embedded_genome_IDs(embeddings_folder_path)
    embedded_text_IDs, text_ID_to_species_first_name_dict = get_embedded_genome_IDs(text_embeddings_folder_path)
    Evo_MIC_count_file_path = current_directory / 'DataPrepare' / 'Data' / 'Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json'

    # original_names_with_genome_embedding: 那些有对应 Evo2 embedding 的 DNAASP 中的完整 strain name
    original_names_with_genome_embedding_handcrafted, original_names_with_genome_embedding_DBAASP_original, origin_to_standard_name_map_dict = get_original_strain_name_with_genome_embedding(Evo_MIC_count_file_path, embedded_genome_IDs)

    # 读取原始 DBAASP 中那些有 MIC 的数据
    Evo_strain_MIC_data_path = current_directory / 'DataPrepare' / 'Data' / 'DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv'  #'DBAASP_id_bact_name_SMILES_MIC_Evo.csv'
    all_Evo_MIC_data = pd.read_csv(Evo_strain_MIC_data_path)
    columns_names = all_Evo_MIC_data.columns
    all_Evo_MIC_data = all_Evo_MIC_data.values

    # 读取原始 small  molecules 中那些有 binary 分类的数据
    Evo_strain_MIC_data_path = current_directory / 'DataPrepare' / 'Data' / 'small_molecule' / 'processed' / 'small_molecule_Evo_binary_data.csv'
    SM_Evo_binary_data = pd.read_csv(Evo_strain_MIC_data_path)
    # columns_names = all_Evo_MIC_data.columns
    SM_Evo_binary_data = SM_Evo_binary_data.values

    # 去掉那些带 'del' 的
    del_excluded_data = []
    for MIC_data_line in tqdm(all_Evo_MIC_data, desc=' removing MIC data with "del" in name '):
        if 'del' not in MIC_data_line[1]:
            del_excluded_data.append(MIC_data_line)
    all_Evo_MIC_data = del_excluded_data

    # filter 一下留下那些有对应 strain 的 genome 的 SMILES -> MIC 对数据
    Evo_MIC_data_with_genome_embedding_handcrafted = []
    for MIC_data_line in tqdm(all_Evo_MIC_data, desc=' retriving MIC data with genome embeddings '):
        if MIC_data_line[1] in original_names_with_genome_embedding_handcrafted:
            Evo_MIC_data_with_genome_embedding_handcrafted.append(MIC_data_line)

    Evo_MIC_data_with_genome_embedding_DBAASP_origianl = []
    for MIC_data_line in tqdm(all_Evo_MIC_data, desc=' retriving MIC data with genome embeddings '):
        if MIC_data_line[1] in original_names_with_genome_embedding_DBAASP_original:
            Evo_MIC_data_with_genome_embedding_DBAASP_origianl.append(MIC_data_line)

    #TODO: 这里把那些没有 genome embedding 但是有 text embedding 的读进来然后过滤到 Evo_MIC_data_wo_genome_embedding = []
    #      记得还要过滤掉那些 MIC 对应的strain名字里第二个是 sp. ssp. group 的数据行

    # 加载所有 text embedding，格式为 {orignal strain_ID: text embeddings, ...}
    text_embeddings_wo_genome_dict = load_text_wo_genome_embeddings(text_embeddings_wo_genome_folder_path, text_embedding_scale_factor, device, 'text (without corresponding genome)')
    Evo_MIC_data_wo_genome_embedding = []
    for MIC_data_line in tqdm(all_Evo_MIC_data, desc=' retriving MIC data with only text embeddings '):
        if len(MIC_data_line[1].split(' ')) <= 1:
            continue
        # 调试用
        # if MIC_data_line[1].split(' ')[1] in ['sp.', 'spp.', 'group']:
        #     print(1)
        if MIC_data_line[1].split(' ')[1] not in ['sp.', 'spp.', 'group'] and MIC_data_line[1] in list(text_embeddings_wo_genome_dict.keys()):
            Evo_MIC_data_wo_genome_embedding.append(MIC_data_line)


    # 去掉那些原始 DBAASP 中连 species name 和 ATCC ID 都对不上的数据
    Evo_MIC_data_with_genome_embedding_DBAASP_origianl = np.array(Evo_MIC_data_with_genome_embedding_DBAASP_origianl)
    Evo_MIC_data_with_genome_embedding_DBAASP_origianl = exclude_wrong_species_ATCC_map(Evo_MIC_data_with_genome_embedding_DBAASP_origianl, genome_ID_to_species_first_name_dict)

    # 手动标注过 "*" 的和没有手动标注过 "*" 的处理完之后重新拼接到一起
    Evo_MIC_data_with_genome_embedding = np.concatenate((np.array(Evo_MIC_data_with_genome_embedding_handcrafted), Evo_MIC_data_with_genome_embedding_DBAASP_origianl))

    # 保存一下
    Evo_MIC_data_with_genome_embedding = pd.DataFrame(Evo_MIC_data_with_genome_embedding, columns=columns_names)
    Evo_MIC_data_with_genome_embedding.to_csv(current_directory / 'DataPrepare' / 'Data' / 'DBAASP_id_bact_name_SMILES_MIC_Evo_with_genome.csv', index=False)

    Evo_MIC_data_with_genome_embedding = Evo_MIC_data_with_genome_embedding.values

    # 把这些有 genome embedding 的数据中的 DBAASP 中原始 strain name 替换成 ATCC ID 方便加载 Evo2 embedding
    Evo_MIC_data_with_genome_embedding_standard_name = []
    for line in Evo_MIC_data_with_genome_embedding:
        # 替换原始的 strain name 到 ATCC 或者是下载的genome 的 name，方便embedding载入
        line[1] = origin_to_standard_name_map_dict[line[1]]
        Evo_MIC_data_with_genome_embedding_standard_name.append(line)
    Evo_MIC_data_with_genome_embedding_standard_name = np.array(Evo_MIC_data_with_genome_embedding_standard_name)

    # Evo_MIC_data_with_genome_embedding_standard_name 是那些有 genome 和 text embedding 的 strain 的 MIC data. 现在把他们和只有 text embedding 的 data 拼起来
    Evo_MIC_data_with_genome_or_text_embedding = np.concatenate((Evo_MIC_data_with_genome_embedding_standard_name, np.array(Evo_MIC_data_wo_genome_embedding)))

    # 加载所有 Evo2 genome embedding，格式为 {strain_ID: Evo2 genome embeddings, ...}
    embeddings_dict = load_all_genome_embeddings(embeddings_folder_path, genome_embedding_scale_factor, device, 'genome')
    text_embeddings_dict = load_all_genome_embeddings(text_embeddings_folder_path, text_embedding_scale_factor, device, 'text (with corresponding genome)')

    # TODO: 这里分成两个数据集做了，利用 set 获取全部的 MIC 数据中有哪些 strain name，并将每一种 strain name 的数据分组（dict）保存方便分割数据集
    all_name_set = set(Evo_MIC_data_with_genome_or_text_embedding[:, 1])  # TODO: 这里已经不仅仅是 ATCC 的 ID 和 #001 之类的了，还有只有 text 的那些完全没处理过的 strain name
    all_strain_line_group_dict = {}
    for standard_strain_ID in tqdm(all_name_set, desc=' Getting strain MIC groups, text only'):
        indices = np.where(Evo_MIC_data_with_genome_or_text_embedding[:, 1] == standard_strain_ID)[0]
        all_strain_line_group_dict[standard_strain_ID] = Evo_MIC_data_with_genome_or_text_embedding[indices]

    # 利用 set 获取 MIC 数据中有哪些 ATCC ID，并将每一种 ATCC ID 的数据分组（dict）保存方便分割数据集
    all_standard_name_set = set(Evo_MIC_data_with_genome_embedding_standard_name[:, 1])
    standard_strain_line_group_dict = {}
    for standard_strain_ID in tqdm(all_standard_name_set, desc=' Getting strain MIC groups, genome and text'):
        indices = np.where(Evo_MIC_data_with_genome_embedding_standard_name[:, 1] == standard_strain_ID)[0]
        standard_strain_line_group_dict[standard_strain_ID] = Evo_MIC_data_with_genome_embedding_standard_name[indices]

    # 利用 set 获取 binary 数据中有哪些 strain ID，并将每一种 strain ID 的数据分组（dict）保存方便分割数据集
    SM_all_name_set = set(SM_Evo_binary_data[:, 1])
    SM_all_strain_line_group_dict = {}
    for standard_strain_ID in tqdm(SM_all_name_set, desc=' Getting strain MIC groups, small molecules'):
        indices = np.where(SM_Evo_binary_data[:, 1] == standard_strain_ID)[0]
        SM_all_strain_line_group_dict[standard_strain_ID] = SM_Evo_binary_data[indices]

    # TODO: 根据树的聚类结果把 ATCC number 分类整理好就行，train 和 test 分开
    # all_grouped_species 只包含 species 的 name，没有 strain number
    # all_grouped_species = cluster_species(current_directory/'DataPrepare'/'Data'/'Genome'/'visualization'/'All_species_gt_Taxonomy_Tree_cluster.phy', # TODO: 分 11 类的时候记得一定要把这个树换成 cluster 版本
    #                                       current_directory/'DataPrepare'/'Data'/'Genome'/'old_to_new_NCBI_taxonomy.json',
    #                                       num_clusters=num_clusters)
    ATCC_ID_to_species_name_map_dict, species_name_ATCC_IDs_map_dict = get_ATCC_ID_to_species_name_map(current_directory/'DataPrepare'/'Data'/'Genome'/'ATCC')
    strain_name_to_original_species_names_map, original_species_names_to_strain_name_map = get_original_strain_ID_to_species_name_map(current_directory/'DataPrepare'/'Data'/'Text_Description'/'wo_ATCC'/'embeddings')

    # 把两个 species 到 strain IDs 的 dict 融合一下
    merged_species_name_to_strain_name_map = merge_dict(species_name_ATCC_IDs_map_dict, original_species_names_to_strain_name_map)

    # 修正新旧命名
    with open(current_directory/'DataPrepare'/'Data'/'Genome'/'old_to_new_NCBI_taxonomy.json', 'r', encoding='utf-8') as f:
        old_to_new_NCBI_taxonomy_map = json.load(f)
    new_to_old_NCBI_taxonomy_map = {value:key for key, value in old_to_new_NCBI_taxonomy_map.items()}
    two_way_taxonomy_map = new_to_old_NCBI_taxonomy_map | old_to_new_NCBI_taxonomy_map

    # grouped strains 就是 这个 species 对应的那些 strain number 放一起, 而且这里的 strain 是我们用到的所有 strain
    # all_grouped_strains = []
    # for grouped_species in tqdm(all_grouped_species, desc=' Grouping strains by clustered species'):
    #     grouped_strains = []
    #     for species in grouped_species:
    #
    #         if species in two_way_taxonomy_map.keys():
    #             _strains_1 = merged_species_name_to_strain_name_map.get(species, None)
    #             _strains_2 = merged_species_name_to_strain_name_map.get(two_way_taxonomy_map[species], None)
    #             if _strains_1 is not None:
    #                 grouped_strains.extend(_strains_1)
    #             if _strains_2 is not None:
    #                 grouped_strains.extend(_strains_2)
    #
    #         else:
    #             grouped_strains.extend(merged_species_name_to_strain_name_map[species])
    #
    #     all_grouped_strains.append(grouped_strains)
    #
    # # 固定 group 的顺序方便在不同的显卡上并行
    # combined = list(zip(all_grouped_species, all_grouped_strains))
    # combined_sorted = sorted(combined, key=lambda x: len(x[0]))
    # all_grouped_species, all_grouped_strains = zip(*combined_sorted)
    # all_grouped_species = list(all_grouped_species)
    # all_grouped_strains = list(all_grouped_strains)
    #
    # group_names = ['Serinales', 'Betaproteobacteria', 'FCB', 'VPC', 'BFSP', 'Eurotiomycetes', 'MA', 'Bacillales', 'Enterobacterales', 'Lactobacillales', 'ALs']  # 6, 9, 10, 12, 13, 14, 30, 31, 32, 33, 41 缩写字母代表

    # 这里 group 的都是 strain, 最后一个 [] 用来 test on 那些只有一个 strain 的 species，因为他们从来没有被 test 过，实际的 fold 数量是 len(train_groups) - 1
    # train_groups = [[], [], []]
    # test_groups = [[], [], []]
    train_groups = [[]]

    repeated_speceis_name_NCBI = []

    for species_name, corresponding_ATCC_IDs in merged_species_name_to_strain_name_map.items():

        # 如果重复的已经处理过了就跳过
        if species_name in repeated_speceis_name_NCBI:
            continue

        mergred_corresponding_ATCC_IDs = corresponding_ATCC_IDs

        if species_name in two_way_taxonomy_map.keys():
            # 防止在此处理相同的 strains
            repeated_speceis_name_NCBI.append(two_way_taxonomy_map[species_name])
            _strains_2 = merged_species_name_to_strain_name_map.get(two_way_taxonomy_map[species_name], None)
            if _strains_2 is not None:
                mergred_corresponding_ATCC_IDs.extend(_strains_2)

        train_groups[0].extend(mergred_corresponding_ATCC_IDs)

            # mergred_corresponding_ATCC_IDs.sort()
            # if len(mergred_corresponding_ATCC_IDs) >= 6:
            #     mergred_corresponding_ATCC_IDs[1], mergred_corresponding_ATCC_IDs[2] = mergred_corresponding_ATCC_IDs[2], mergred_corresponding_ATCC_IDs[1]  # 交换第1，2个防止数据量多的都在前面

            # 只有 1 个 strain 的 species 全部放到这里的训练集里
            # if len(mergred_corresponding_ATCC_IDs) == 1:
            #     train_groups[i].extend(mergred_corresponding_ATCC_IDs)
            # elif len(mergred_corresponding_ATCC_IDs) == 2:
            #     train_groups[i].append(mergred_corresponding_ATCC_IDs[i % 2])
            #     test_groups[i].append(mergred_corresponding_ATCC_IDs[(i + 1) % 2])
            # else:
            #     # chunk_length = len(corresponding_ATCC_IDs) // (len(train_groups) - 1)  # 注意这个是真正的 fold 大小
            #     chunk_length = len(mergred_corresponding_ATCC_IDs) // len(train_groups)
            #     chunked_ATCC_IDs_for_test = mergred_corresponding_ATCC_IDs[i * chunk_length: (i + 1) * chunk_length]
            #     chunked_ATCC_IDs_for_train = list(set(mergred_corresponding_ATCC_IDs) - set(chunked_ATCC_IDs_for_test))
            #     train_groups[i].extend(chunked_ATCC_IDs_for_train)
            #     test_groups[i].extend(chunked_ATCC_IDs_for_test)

    group_names = ['fold 1']#, 'fold 2', 'fold 3']

    #TODO: 调试用
    # args.parallel = True
    # args.test_group = 1


    # 循环测试所有的 group
    for i, (strain_for_train, test_group_name) in enumerate(zip(train_groups, group_names)):

        # 如果要 parallel 地 validate，那么在当前 group 不是 目标 test group 的时候直接跳过
        if args.parallel:
            # if test_group_name != args.test_group:
            if i != args.test_group:
                continue
        # print(f'\n Current test group: {test_group_name}\n')
        # logger.info(f'\n Current test group: {test_group_name}\n')

        # strain_for_test = ['11060', '29930']  # 19417
        # gt_strain_for_test = set(strain_for_test) & all_standard_name_set
        gt_strain_for_train = set(strain_for_train) & all_standard_name_set  # all_standard_name_set - gt_strain_for_test

        gt_train_data = []
        # gt_test_data = []

        for strain_ID in gt_strain_for_train:
            gt_train_data.append(standard_strain_line_group_dict[strain_ID])
        # for strain_ID in gt_strain_for_test:
        #     gt_test_data.append(standard_strain_line_group_dict[strain_ID])

        gt_train_data = pd.DataFrame(np.concatenate(gt_train_data), columns=columns_names)
        # gt_test_data = pd.DataFrame(np.concatenate(gt_test_data), columns=columns_names)

        gt_train_mean_MIC = -np.log10(gt_train_data['MIC'].mean()/10)

        # 新开一个只有 text embedding 的 dataset, 这里 test set 只留下那些和 genome text 都有的 test set 中不重合的的 strain，以免重复计算 test
        # t_strain_for_test = (set(strain_for_test) & all_name_set) - gt_strain_for_test
        t_strain_for_train = set(strain_for_train) & all_name_set  # all_name_set - t_strain_for_test - gt_strain_for_test

        t_train_data = []
        # t_test_data = []

        for strain_ID in t_strain_for_train:
            t_train_data.append(all_strain_line_group_dict[strain_ID])
        # for strain_ID in t_strain_for_test:
        #     t_test_data.append(all_strain_line_group_dict[strain_ID])

        t_train_data = pd.DataFrame(np.concatenate(t_train_data), columns=columns_names)
        # t_test_data = pd.DataFrame(np.concatenate(t_test_data), columns=columns_names)

        t_train_mean_MIC = -np.log10(t_train_data['MIC'].mean() / 10)

        # SM_gt_train_data = []
        # if '#004' in gt_strain_for_train:
        #     SM_gt_train_data.extend([line for line in SM_Evo_binary_data if line[1] == '#004'])
        # if '17978' in gt_strain_for_train:
        #     SM_gt_train_data.extend([line for line in SM_Evo_binary_data if line[1] == '17978'])
        #
        # if len(SM_gt_train_data) > 0:
        #     SM_gt_train_data = pd.DataFrame(SM_gt_train_data, columns=columns_names)
        #
        # SM_t_train_data = []
        # if 'Staphylococcus aureus RN4220' in t_strain_for_train:
        #     SM_t_train_data.extend([line for line in SM_Evo_binary_data if line[1] == 'Staphylococcus aureus RN4220'])
        #
        # if len(SM_t_train_data) > 0:
        #     SM_t_train_data = pd.DataFrame(SM_t_train_data, columns=columns_names)

        model_name = "ibm-research/materials.selfies-ted"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set global tokenizer for collate functions
        set_tokenizer(tokenizer)

        gt_train_dataset = SMILESDataset_with_genome_and_text(gt_train_data, tokenizer, embeddings_dict, text_embeddings_dict, 'peptide genome-text training set')
        # gt_test_dataset = SMILESDataset_with_genome_and_text(gt_test_data, tokenizer, embeddings_dict, text_embeddings_dict, 'peptide genome-text test set')

        # 分别加载了两个 text embedding dict, 一个是只有 text embedding 的，还有一个是 genome 和 text embedding 都有的
        all_text_embedding_dict = text_embeddings_dict | text_embeddings_wo_genome_dict

        t_train_dataset = SMILESDataset_with_text_only(t_train_data, tokenizer, all_text_embedding_dict, 'peptide text-only training set')
        # t_test_dataset = SMILESDataset_with_text_only(t_test_data, tokenizer, all_text_embedding_dict,'peptide text-only test set')

        # if len(SM_gt_train_data) > 0:
        #     SM_gt_train_dataset = SMILESDataset_with_genome_and_text(SM_gt_train_data, tokenizer, embeddings_dict, text_embeddings_dict, 'small-molecule genome-text training set')
        #
        # if len(SM_t_train_data) > 0:
        #     SM_t_train_dataset = SMILESDataset_with_text_only(SM_t_train_data, tokenizer, all_text_embedding_dict, 'small-molecule text-only training set')

        # print(f"\n training data 1 data type: {gt_train_dataset[0]['genome_embedding'].dtype}\n")
        logger.info(f"\n training data 1 data type: {gt_train_dataset[0]['genome_embedding'].dtype}\n")

        # ========================================================================
        # ENSEMBLE TRAINING LOOP
        # ========================================================================
        test_predictions_of_ensembles = []
        for ensemble in tqdm(range(num_ensembles), desc=' Doing ensembles '):
            # 设置对应的 随机数种子
            torch.manual_seed(random_seeds[ensemble])
            torch.cuda.manual_seed(random_seeds[ensemble])

            # --------------------------------------------------------------------
            # Hyperparameters
            # --------------------------------------------------------------------
            num_epochs = args.epoch
            min_lr = 1e-10
            batch_size = 70
            freeze_epochs = 5000

            logger.info(f' num of frozen epochs: {freeze_epochs}\n')

            # --------------------------------------------------------------------
            # Model Initialization
            # --------------------------------------------------------------------
            DIT_ckpt_path = '/data2/tianang/projects/mdlm/Checkpoints_fangping/last_reg_v1.ckpt'
            mdlm_model = mol_emb_mdlm(config, len(tokenizer.get_vocab()), DIT_ckpt_path, tokenizer.mask_token_id)
            mdlm_model.to(device)
            mdlm_model.eval()
            # DIT = load_DIT(len(tokenizer.get_vocab()), DIT_ckpt_path)

            # 冻结预训练模型参数
            for param in mdlm_model.parameters():
                param.requires_grad = False

            genome_dim = gt_train_dataset[0]['genome_embedding'].shape[1]
            text_dim = gt_train_dataset[0]['text_embedding'].shape[1]

            co_cross_attn_genome = FirstTokenAttention_genome(mdlm_model.config.model.hidden_size, gt_train_dataset[0]['genome_embedding'].shape[1], 4, 0.1)
            co_cross_attn_genome.to(device)
            co_cross_attn_text = FirstTokenAttention_genome(mdlm_model.config.model.hidden_size, gt_train_dataset[0]['text_embedding'].shape[1], 4, 0.1)
            co_cross_attn_text.to(device)
            reg_head = RegressionHead(genome_dim + text_dim, (genome_dim + text_dim)//4, 128, 1, 0.2)
            reg_head.to(device)
            cls_head = RegressionHead(genome_dim + text_dim, (genome_dim + text_dim) // 4, 128, 1, 0.2)
            cls_head.to(device)

            learnable_embedding_weight = nn.Parameter(torch.randn(1, genome_dim, device=device))

            criterion = nn.MSELoss()
            cls_criterion = nn.BCEWithLogitsLoss()
            scaler = torch.cuda.amp.GradScaler()
            # scaler = torch.amp.GradScaler('cuda')
            optimizer = optim.Adam(co_cross_attn_genome.parameters(), lr=1e-5, weight_decay=args.weight_decay)  # 1e-5
            optimizer.add_param_group({'params': co_cross_attn_text.parameters(), 'lr': 1e-5, 'weight_decay': args.weight_decay})
            optimizer.add_param_group({'params': reg_head.parameters(), 'lr': 1e-5, 'weight_decay': args.weight_decay})
            optimizer.add_param_group({'params': cls_head.parameters(), 'lr': 1e-5, 'weight_decay': args.weight_decay})
            optimizer.add_param_group({'params': [learnable_embedding_weight], 'lr': 1e-5, 'weight_decay': args.weight_decay})
            optimizer.add_param_group({'params': mdlm_model.parameters(), 'lr': 3e-6, 'weight_decay': args.weight_decay*0.1})   # TODO: ChemBERTa 和别的学习率不一样

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

            gt_train_loader = DataLoader(gt_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            # gt_test_loader = DataLoader(gt_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            t_train_loader = DataLoader(t_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_text_only)
            # t_test_loader = DataLoader(t_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_text_only)

            # if len(SM_gt_train_data) > 0:
            #     SM_gt_train_loader = DataLoader(SM_gt_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_cls)
            # else:
            #     SM_gt_train_loader = [None]
            # # SM_gt_test_loader = DataLoader(SM_gt_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            #
            # if len(SM_t_train_data) > 0:
            #     SM_t_train_loader = DataLoader(SM_t_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_text_only_cls)
            # else:
            #     SM_t_train_loader = [None]

            best_R2_test = -10
            best_spearman_test = -10
            best_pearson_test = -10
            best_test_prdictions = None
            for epoch in tqdm(range(num_epochs), desc=f' Training ensemble {ensemble+1}/{num_ensembles} ', leave=False):

                if epoch == freeze_epochs:
                    # 解冻预训练模型
                    for param in mdlm_model.parameters():
                        param.requires_grad = True
                    # optimizer.add_param_group({'params': ChemBERTa_model.parameters(), 'lr': 1e-7})  # 在这里加的话会导致 scheduler 里面没有 ChemBERTa 的权重
                    # print(f'\n ChemBERTa now open for training')
                    logger.info(f'\n\n ChemBERTa now open for training')

                # 查看随机初始化状态下测试集的 R2 能到多少
    #             if epoch == 0:
    #                 with torch.no_grad():
    #
    #                     test_batch_losses = []
    #                     test_all_labels = []
    #                     test_all_preds = []
    #                     species_wise_test_labels_dict = {}
    #                     species_wise_test_preds_dict = {}
    #
    #                     gt_test_batch_losses = []
    #                     gt_test_all_labels = []
    #                     gt_test_all_preds = []
    #                     t_test_batch_losses = []
    #                     t_test_all_labels = []
    #                     t_test_all_preds = []
    #                     train_mean_as_test_predict = []
    #
    #                     for gt_batch, t_batch in tqdm(itertools.zip_longest(gt_test_loader, t_test_loader, fillvalue=None), desc=f" Epoch {epoch}/{num_epochs} | evaluating", leave=False, total=max(len(gt_test_loader), len(t_test_loader))):
    #                         if gt_batch is not None:
    #                             input_ids = gt_batch['input_ids'].to(device)
    #                             attention_mask = gt_batch['attention_mask'].to(device)
    #                             labels = gt_batch['label'].to(device)
    #                             padded_genome_embeddings = gt_batch['padded_genome_embeddings']  # .to(torch.float)
    #                             genome_attn_masks = gt_batch['genome_attn_masks']
    #                             padded_text_embeddings = gt_batch['padded_text_embeddings']  # .to(torch.float)
    #                             text_attn_masks = gt_batch['text_attn_masks']
    #                             strain_names = gt_batch['strain_names']
    #
    #                             with torch.amp.autocast('cuda', enabled=True):
    #                                 outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
    #
    #                                 mol_cls_embedding = outputs[:, 0, :]
    #                                 mol_cls_embedding_genome = co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
    #                                 mol_cls_embedding_text = co_cross_attn_text(mol_cls_embedding, padded_text_embeddings,1 - text_attn_masks)
    #                                 mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
    #                                 logits = reg_head(mol_cls_embedding)
    #                                 loss = criterion(logits.squeeze(), labels.squeeze())
    #
    #                             test_batch_losses.append(loss.item())
    #                             gt_test_batch_losses.append(loss.item())
    #
    #                             test_batch_labels = labels.detach().cpu().flatten().tolist()
    #                             test_batch_preds = logits.detach().cpu().flatten().tolist()
    #
    #                             test_all_labels.extend(test_batch_labels)
    #                             test_all_preds.extend(test_batch_preds)
    #                             gt_test_all_labels.extend(test_batch_labels)
    #                             gt_test_all_preds.extend(test_batch_preds)
    #                             train_mean_as_test_predict.extend(np.full(logits.detach().cpu().flatten().shape, gt_train_mean_MIC).tolist())
    #
    #                             for strain_name, label, pred in zip(strain_names, test_batch_labels, test_batch_preds):
    #                                 _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
    #                                 if _speceis_name is None:
    #                                     _speceis_name = strain_name_to_original_species_names_map[strain_name]
    #                                 if _speceis_name not in species_wise_test_preds_dict.keys():
    #                                     species_wise_test_preds_dict[_speceis_name] = [pred]
    #                                     species_wise_test_labels_dict[_speceis_name] = [label]
    #                                 else:
    #                                     species_wise_test_preds_dict[_speceis_name].append(pred)
    #                                     species_wise_test_labels_dict[_speceis_name].append(label)
    #
    #                         if t_batch is not None:
    #                             input_ids = t_batch['input_ids'].to(device)
    #                             attention_mask = t_batch['attention_mask'].to(device)
    #                             labels = t_batch['label'].to(device)
    #                             # padded_genome_embeddings = gt_batch['padded_genome_embeddings']  # .to(torch.float)
    #                             # genome_attn_masks = gt_batch['genome_attn_masks']
    #                             padded_text_embeddings = t_batch['padded_text_embeddings']  # .to(torch.float)
    #                             text_attn_masks = t_batch['text_attn_masks']
    #                             strain_names = t_batch['strain_names']
    #
    #                             with torch.amp.autocast('cuda', enabled=True):
    #                                 outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
    #
    #                                 mol_cls_embedding = outputs[:, 0, :]
    #                                 padded_genome_embeddings = learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
    #                                 genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1).to(device)
    #                                 mol_cls_embedding_genome = co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
    #                                 # 把 learnable embedding 的 batch 纬 expand 作为 genome embedding 的替换
    #                                 mol_cls_embedding_text = co_cross_attn_text(mol_cls_embedding, padded_text_embeddings,1 - text_attn_masks)
    #                                 mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
    #                                 logits = reg_head(mol_cls_embedding)
    #                                 loss = criterion(logits.squeeze(), labels.squeeze())
    #
    #                             test_batch_losses.append(loss.item())
    #                             t_test_batch_losses.append(loss.item())
    #
    #                             test_batch_labels = labels.detach().cpu().flatten().tolist()
    #                             test_batch_preds = logits.detach().cpu().flatten().tolist()
    #
    #                             test_all_labels.extend(test_batch_labels)
    #                             test_all_preds.extend(test_batch_preds)
    #                             t_test_all_labels.extend(test_batch_labels)
    #                             t_test_all_preds.extend(test_batch_preds)
    #                             train_mean_as_test_predict.extend(np.full(logits.detach().cpu().flatten().shape, t_train_mean_MIC).tolist())
    #
    #                             for strain_name, label, pred in zip(strain_names, test_batch_labels, test_batch_preds):
    #                                 _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
    #                                 if _speceis_name is None:
    #                                     _speceis_name = strain_name_to_original_species_names_map[strain_name]
    #                                 if _speceis_name not in species_wise_test_preds_dict.keys():
    #                                     species_wise_test_preds_dict[_speceis_name] = [pred]
    #                                     species_wise_test_labels_dict[_speceis_name] = [label]
    #                                 else:
    #                                     species_wise_test_preds_dict[_speceis_name].append(pred)
    #                                     species_wise_test_labels_dict[_speceis_name].append(label)
    #
    #                     r2 = calculate_r2(test_all_labels, test_all_preds)
    #                     gt_r2 = calculate_r2(gt_test_all_labels, gt_test_all_preds)
    #                     t_r2 = calculate_r2(t_test_all_labels, t_test_all_preds)
    #                     r2_train_mean = calculate_r2(test_all_labels, train_mean_as_test_predict)
    #
    #                     r2_MSE_spearman_pearson_species_wise = {}
    #                     for _speceis_name in species_wise_test_preds_dict.keys():
    #                         r2_species = calculate_r2(species_wise_test_labels_dict[_speceis_name], species_wise_test_preds_dict[_speceis_name])
    #                         MSE_specise = np.mean((np.array(species_wise_test_labels_dict[_speceis_name]) - np.array(species_wise_test_preds_dict[_speceis_name])) ** 2)
    #                         if len(species_wise_test_labels_dict[_speceis_name]) > 1:
    #                             spearman_species = spearmanr(species_wise_test_labels_dict[_speceis_name],
    #                                                          species_wise_test_preds_dict[_speceis_name])[0]
    #                             pearson_species = pearsonr(species_wise_test_labels_dict[_speceis_name],
    #                                                        species_wise_test_preds_dict[_speceis_name])[0]
    #                         else:
    #                             spearman_species = pearson_species = None
    #                         r2_MSE_spearman_pearson_species_wise[_speceis_name] = [r2_species, MSE_specise, spearman_species, pearson_species]
    #
    #                     logger.info(f'\n Test species wise R2, MSE, Spearman, Pearson:')
    #                     for species_name, metrics in r2_MSE_spearman_pearson_species_wise.items():
    #                         formatted_metrics = ", ".join(f"{m:.4f}" if isinstance(m, float) else str(m) for m in metrics)
    #                         logger.info(f'    {species_name}:  {formatted_metrics}')
    #
    # #                     print(f""" Ensemble {ensemble+1}/{num_ensembles} Epoch {epoch}/{num_epochs}
    # # Test Loss: {np.array(test_batch_losses).mean():.6f}, genome text Test Loss: {np.array(gt_test_batch_losses).mean():.6f}, text only Test Loss: {np.array(t_test_batch_losses).mean():.6f}
    # # Test R2: {r2:.6f}, genome text Test R2: {gt_r2:.6f}, text only Test R2: {t_r2:.6f}, Test train mean MIC R2: {r2_train_mean:.6f}""")
    #                     logger.info(f""" Ensemble {ensemble + 1}/{num_ensembles} Epoch {epoch}/{num_epochs}
    # Test Loss: {np.array(test_batch_losses).mean():.6f}, genome text Test Loss: {np.array(gt_test_batch_losses).mean():.6f}, text only Test Loss: {np.array(t_test_batch_losses).mean():.6f}
    # Test R2: {r2:.6f}, genome text Test R2: {gt_r2:.6f}, text only Test R2: {t_r2:.6f}, Test train mean MIC R2: {r2_train_mean:.6f}""")

                train_batch_losses = []
                train_all_labels = []
                train_all_preds = []
                gt_train_batch_losses = []
                gt_train_all_labels = []
                gt_train_all_preds = []
                t_train_batch_losses = []
                t_train_all_labels = []
                t_train_all_preds = []

                # species_wise_train_labels_dict = {}
                # species_wise_train_preds_dict = {}
                #
                # cls_train_batch_losses = []
                # cls_gt_train_batch_losses = []
                #
                # cls_train_all_labels = []
                # cls_train_all_preds = []
                # cls_gt_train_all_labels = []
                # cls_gt_train_all_preds = []
                # cls_t_train_batch_losses = []
                # cls_t_train_all_labels = []
                # cls_t_train_all_preds = []

                # for gt_batch, t_batch, SM_gt_batch, SM_t_batch in tqdm(itertools.zip_longest(gt_train_loader, t_train_loader, SM_gt_train_loader, SM_t_train_loader, fillvalue=None), desc=f" Ensemble {ensemble + 1}/{num_ensembles} Epoch {epoch + 1}/{num_epochs} | training", leave=False, total=max(len(gt_train_loader), len(t_train_loader), len(SM_gt_train_loader), len(SM_t_train_loader))):
                for gt_batch, t_batch in tqdm(
                        itertools.zip_longest(gt_train_loader, t_train_loader, fillvalue=None),
                        desc=f" Ensemble {ensemble + 1}/{num_ensembles} Epoch {epoch + 1}/{num_epochs} | training",
                        leave=False, total=max(len(gt_train_loader), len(t_train_loader))):

                    if gt_batch is not None:
                        input_ids = gt_batch['input_ids'].to(device)
                        attention_mask = gt_batch['attention_mask'].to(device)
                        labels = gt_batch['label'].to(device)
                        padded_genome_embeddings = gt_batch['padded_genome_embeddings']  # .to(torch.float)
                        genome_attn_masks = gt_batch['genome_attn_masks']
                        padded_text_embeddings = gt_batch['padded_text_embeddings']  # .to(torch.float)
                        text_attn_masks = gt_batch['text_attn_masks']
                        strain_names = gt_batch['strain_names']

                        optimizer.zero_grad()

                        with torch.amp.autocast('cuda', enabled=True):
                            outputs = mdlm_model(input_ids=input_ids, attention_mask=attention_mask)

                            mol_cls_embedding = outputs[:, 0, :]
                            mol_cls_embedding_genome, _ = co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
                            mol_cls_embedding_text, _ = co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
                            mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
                            logits = reg_head(mol_cls_embedding).squeeze()
                            loss = criterion(logits, labels.squeeze())

                        # loss.backward()
                        # optimizer.step()

                        scaler.scale(loss).backward()
                        # 对模型参数的梯度进行裁剪，例如设置最大范数为 1.0
                        if epoch >= freeze_epochs:
                            # 将梯度 unscale 到正常范围
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(mdlm_model.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(co_cross_attn_genome.parameters(), max_norm=1.0)
                            # torch.nn.utils.clip_grad_norm_(co_cross_attn_text.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(reg_head.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                        train_batch_losses.append(loss.item())
                        gt_train_batch_losses.append(loss.item())

                        train_batch_labels = labels.detach().cpu().flatten().tolist()
                        train_batch_preds = logits.detach().cpu().flatten().tolist()

                        train_all_labels.extend(train_batch_labels)
                        train_all_preds.extend(train_batch_preds)
                        gt_train_all_labels.extend(train_batch_labels)
                        gt_train_all_preds.extend(train_batch_preds)

                        # for strain_name, label, pred in zip(strain_names, train_batch_labels, train_batch_preds):
                        #     _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
                        #     if _speceis_name is None:
                        #         _speceis_name = strain_name_to_original_species_names_map[strain_name]
                        #     if _speceis_name not in species_wise_train_preds_dict.keys():
                        #         species_wise_train_preds_dict[_speceis_name] = [pred]
                        #         species_wise_train_labels_dict[_speceis_name] = [label]
                        #     else:
                        #         species_wise_train_preds_dict[_speceis_name].append(pred)
                        #         species_wise_train_labels_dict[_speceis_name].append(label)

                    if t_batch is not None:
                        input_ids = t_batch['input_ids'].to(device)
                        attention_mask = t_batch['attention_mask'].to(device)
                        labels = t_batch['label'].to(device)
                        # padded_genome_embeddings = t_batch['padded_genome_embeddings']  # .to(torch.float)
                        # genome_attn_masks = t_batch['genome_attn_masks']
                        padded_text_embeddings = t_batch['padded_text_embeddings']  # .to(torch.float)
                        text_attn_masks = t_batch['text_attn_masks']
                        strain_names = t_batch['strain_names']

                        optimizer.zero_grad()

                        with torch.amp.autocast('cuda', enabled=True):
                            outputs = mdlm_model(input_ids=input_ids, attention_mask=attention_mask)

                            mol_cls_embedding = outputs[:, 0, :]
                            padded_genome_embeddings = learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
                            genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1).to(device)
                            mol_cls_embedding_genome, _ = co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
                            # 把 learnable embedding 的 batch 纬 expand 作为 genome embedding 的替换
                            # mol_cls_embedding_genome = learnable_embedding_weight.expand(mol_cls_embedding.shape[0], -1)
                            mol_cls_embedding_text, _ = co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
                            mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
                            logits = reg_head(mol_cls_embedding).squeeze()
                            loss = criterion(logits, labels.squeeze())

                        # loss.backward()
                        # optimizer.step()

                        scaler.scale(loss).backward()
                        # 对模型参数的梯度进行裁剪，例如设置最大范数为 1.0
                        if epoch >= freeze_epochs:
                            # 将梯度 unscale 到正常范围
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(mdlm_model.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_([learnable_embedding_weight], max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(co_cross_attn_genome.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(reg_head.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                        train_batch_losses.append(loss.item())
                        t_train_batch_losses.append(loss.item())

                        train_batch_labels = labels.detach().cpu().flatten().tolist()
                        train_batch_preds = logits.detach().cpu().flatten().tolist()

                        train_all_labels.extend(train_batch_labels)
                        train_all_preds.extend(train_batch_preds)
                        t_train_all_labels.extend(train_batch_labels)
                        t_train_all_preds.extend(train_batch_preds)

                        # for strain_name, label, pred in zip(strain_names, train_batch_labels, train_batch_preds):
                        #     _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
                        #     if _speceis_name is None:
                        #         _speceis_name = strain_name_to_original_species_names_map[strain_name]
                        #     if _speceis_name not in species_wise_train_preds_dict.keys():
                        #         species_wise_train_preds_dict[_speceis_name] = [pred]
                        #         species_wise_train_labels_dict[_speceis_name] = [label]
                        #     else:
                        #         species_wise_train_preds_dict[_speceis_name].append(pred)
                        #         species_wise_train_labels_dict[_speceis_name].append(label)

                    # if SM_gt_batch is not None:
                    #     input_ids = SM_gt_batch['input_ids'].to(device)
                    #     attention_mask = SM_gt_batch['attention_mask'].to(device)
                    #     labels = SM_gt_batch['label'].to(device)
                    #     padded_genome_embeddings = SM_gt_batch['padded_genome_embeddings']  # .to(torch.float)
                    #     genome_attn_masks = SM_gt_batch['genome_attn_masks']
                    #     padded_text_embeddings = SM_gt_batch['padded_text_embeddings']  # .to(torch.float)
                    #     text_attn_masks = SM_gt_batch['text_attn_masks']
                    #     strain_names = SM_gt_batch['strain_names']
                    #
                    #     optimizer.zero_grad()
                    #
                    #     with torch.amp.autocast('cuda', enabled=True):
                    #         outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
                    #
                    #         mol_cls_embedding = outputs[:, 0, :]
                    #         mol_cls_embedding_genome = co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
                    #         mol_cls_embedding_text = co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
                    #         mol_cls_embedding = torch.cat((mol_cls_embedding_genome, mol_cls_embedding_text), dim=1)
                    #         logits = cls_head(mol_cls_embedding).squeeze()
                    #         loss = cls_criterion(logits, labels.squeeze())
                    #
                    #     # loss.backward()
                    #     # optimizer.step()
                    #
                    #     scaler.scale(loss).backward()
                    #     # 对模型参数的梯度进行裁剪，例如设置最大范数为 1.0
                    #     if epoch >= freeze_epochs:
                    #         # 将梯度 unscale 到正常范围
                    #         scaler.unscale_(optimizer)
                    #         torch.nn.utils.clip_grad_norm_(ChemBERTa_model.parameters(), max_norm=1.0)
                    #         torch.nn.utils.clip_grad_norm_(co_cross_attn_genome.parameters(), max_norm=1.0)
                    #         torch.nn.utils.clip_grad_norm_(reg_head.parameters(), max_norm=1.0)
                    #     scaler.step(optimizer)
                    #     scaler.update()
                    #
                    #     cls_train_batch_losses.append(loss.item())
                    #     cls_gt_train_batch_losses.append(loss.item())
                    #
                    #     cls_train_all_labels.extend(labels.detach().cpu().flatten().tolist())
                    #     cls_train_all_preds.extend(logits.detach().cpu().flatten().tolist())
                    #     cls_gt_train_all_labels.extend(labels.detach().cpu().flatten().tolist())
                    #     cls_gt_train_all_preds.extend(logits.detach().cpu().flatten().tolist())
                    #
                    # if SM_t_batch is not None:
                    #     input_ids = SM_t_batch['input_ids'].to(device)
                    #     attention_mask = SM_t_batch['attention_mask'].to(device)
                    #     labels = SM_t_batch['label'].to(device)
                    #     # padded_genome_embeddings = t_batch['padded_genome_embeddings']  # .to(torch.float)
                    #     # genome_attn_masks = t_batch['genome_attn_masks']
                    #     padded_text_embeddings = SM_t_batch['padded_text_embeddings']  # .to(torch.float)
                    #     text_attn_masks = SM_t_batch['text_attn_masks']
                    #     strain_names = SM_t_batch['strain_names']
                    #
                    #     optimizer.zero_grad()
                    #
                    #     with torch.amp.autocast('cuda', enabled=True):
                    #         outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
                    #
                    #         mol_cls_embedding = outputs[:, 0, :]
                    #         padded_genome_embeddings = learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
                    #         genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1).to(device)
                    #         mol_cls_embedding_genome = co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
                    #         # 把 learnable embedding 的 batch 纬 expand 作为 genome embedding 的替换
                    #         # mol_cls_embedding_genome = learnable_embedding_weight.expand(mol_cls_embedding.shape[0], -1)
                    #         mol_cls_embedding_text = co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
                    #         mol_cls_embedding = torch.cat((mol_cls_embedding_genome, mol_cls_embedding_text), dim=1)
                    #         logits = cls_head(mol_cls_embedding).squeeze()
                    #         loss = cls_criterion(logits, labels.squeeze())
                    #
                    #     # loss.backward()
                    #     # optimizer.step()
                    #
                    #     scaler.scale(loss).backward()
                    #     # 对模型参数的梯度进行裁剪，例如设置最大范数为 1.0
                    #     if epoch >= freeze_epochs:
                    #         # 将梯度 unscale 到正常范围
                    #         scaler.unscale_(optimizer)
                    #         torch.nn.utils.clip_grad_norm_(ChemBERTa_model.parameters(), max_norm=1.0)
                    #         torch.nn.utils.clip_grad_norm_([learnable_embedding_weight], max_norm=1.0)
                    #         torch.nn.utils.clip_grad_norm_(co_cross_attn_genome.parameters(), max_norm=1.0)
                    #         torch.nn.utils.clip_grad_norm_(reg_head.parameters(), max_norm=1.0)
                    #     scaler.step(optimizer)
                    #     scaler.update()
                    #
                    #     cls_train_batch_losses.append(loss.item())
                    #     cls_t_train_batch_losses.append(loss.item())
                    #
                    #     cls_train_all_labels.extend(labels.detach().cpu().flatten().tolist())
                    #     cls_train_all_preds.extend(logits.detach().cpu().flatten().tolist())
                    #     cls_t_train_all_labels.extend(labels.detach().cpu().flatten().tolist())
                    #     cls_t_train_all_preds.extend(logits.detach().cpu().flatten().tolist())

                scheduler.step()
                # print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {np.array(batch_losses).mean()}")

                r2_train = calculate_r2(train_all_labels, train_all_preds)
                spearman_train = spearmanr(train_all_labels, train_all_preds)[0]
                pearson_train = pearsonr(train_all_labels, train_all_preds)[0]
                gt_r2_train = calculate_r2(gt_train_all_labels, gt_train_all_preds)
                gt_spearman_train = spearmanr(gt_train_all_labels, gt_train_all_preds)[0]
                gt_pearson_train = pearsonr(gt_train_all_labels, gt_train_all_preds)[0]
                t_r2_train = calculate_r2(t_train_all_labels, t_train_all_preds)
                t_spearman_train = spearmanr(t_train_all_labels, t_train_all_preds)[0]
                t_pearson_train = pearsonr(t_train_all_labels, t_train_all_preds)[0]

                r2_MSE_spearman_pearson_species_wise = {}
                # for _speceis_name in species_wise_train_preds_dict.keys():
                #     r2_species = calculate_r2(species_wise_train_labels_dict[_speceis_name], species_wise_train_preds_dict[_speceis_name])
                #     MSE_specise = np.mean((np.array(species_wise_train_labels_dict[_speceis_name]) - np.array(species_wise_train_preds_dict[_speceis_name])) ** 2)
                #     if len(species_wise_train_labels_dict[_speceis_name]) > 1:
                #         spearman_species = spearmanr(species_wise_train_labels_dict[_speceis_name], species_wise_train_preds_dict[_speceis_name])[0]
                #         pearson_species = pearsonr(species_wise_train_labels_dict[_speceis_name], species_wise_train_preds_dict[_speceis_name])[0]
                #     else:
                #         spearman_species = pearson_species = None
                #     r2_MSE_spearman_pearson_species_wise[_speceis_name] = [r2_species, MSE_specise, spearman_species, pearson_species]

                # logger.info(f'\n Train species wise R2, MSE, Spearman, Pearson:')
                # for species_name, metrics in r2_MSE_spearman_pearson_species_wise.items():
                #     formatted_metrics = ", ".join(f"{m:.4f}" if isinstance(m, float) else str(m) for m in metrics)
                #     logger.info(f'    {species_name}:  {formatted_metrics}')

    #             with torch.no_grad():
    #
    #                 test_batch_losses = []
    #                 test_all_labels = []
    #                 test_all_preds = []
    #                 gt_test_batch_losses = []
    #                 gt_test_all_labels = []
    #                 gt_test_all_preds = []
    #                 t_test_batch_losses = []
    #                 t_test_all_labels = []
    #                 t_test_all_preds = []
    #
    #                 species_wise_test_labels_dict = {}
    #                 species_wise_test_preds_dict = {}
    #
    #                 for gt_batch, t_batch in tqdm(itertools.zip_longest(gt_test_loader, t_test_loader, fillvalue=None), desc=f" Ensemble {ensemble+1}/{num_ensembles} Epoch {epoch + 1}/{num_epochs} | evaluating", leave=False, total=max(len(gt_test_loader), len(t_test_loader))):
    #                     if gt_batch is not None:
    #                         input_ids = gt_batch['input_ids'].to(device)
    #                         attention_mask = gt_batch['attention_mask'].to(device)
    #                         labels = gt_batch['label'].to(device)
    #                         padded_genome_embeddings = gt_batch['padded_genome_embeddings']  # .to(torch.float)
    #                         genome_attn_masks = gt_batch['genome_attn_masks']
    #                         padded_text_embeddings = gt_batch['padded_text_embeddings']  # .to(torch.float)
    #                         text_attn_masks = gt_batch['text_attn_masks']
    #                         strain_names = gt_batch['strain_names']
    #
    #                         with torch.amp.autocast('cuda', enabled=True):
    #                             outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
    #
    #                             mol_cls_embedding = outputs[:, 0, :]
    #                             mol_cls_embedding_genome = co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
    #                             mol_cls_embedding_text = co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
    #                             mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
    #                             logits = reg_head(mol_cls_embedding)
    #                             loss = criterion(logits.squeeze(), labels.squeeze())
    #
    #                         test_batch_losses.append(loss.item())
    #                         gt_test_batch_losses.append(loss.item())
    #
    #                         test_batch_labels = labels.detach().cpu().flatten().tolist()
    #                         test_batch_preds = logits.detach().cpu().flatten().tolist()
    #
    #                         test_all_labels.extend(test_batch_labels)
    #                         test_all_preds.extend(test_batch_preds)
    #                         gt_test_all_labels.extend(test_batch_labels)
    #                         gt_test_all_preds.extend(test_batch_preds)
    #
    #                         for strain_name, label, pred in zip(strain_names, test_batch_labels, test_batch_preds):
    #                             _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
    #                             if _speceis_name is None:
    #                                 _speceis_name = strain_name_to_original_species_names_map[strain_name]
    #                             if _speceis_name not in species_wise_test_preds_dict.keys():
    #                                 species_wise_test_preds_dict[_speceis_name] = [pred]
    #                                 species_wise_test_labels_dict[_speceis_name] = [label]
    #                             else:
    #                                 species_wise_test_preds_dict[_speceis_name].append(pred)
    #                                 species_wise_test_labels_dict[_speceis_name].append(label)
    #
    #                     if t_batch is not None:
    #                         input_ids = t_batch['input_ids'].to(device)
    #                         attention_mask = t_batch['attention_mask'].to(device)
    #                         labels = t_batch['label'].to(device)
    #                         # padded_genome_embeddings = gt_batch['padded_genome_embeddings']  # .to(torch.float)
    #                         # genome_attn_masks = gt_batch['genome_attn_masks']
    #                         padded_text_embeddings = t_batch['padded_text_embeddings']  # .to(torch.float)
    #                         text_attn_masks = t_batch['text_attn_masks']
    #                         strain_names = t_batch['strain_names']
    #
    #                         with torch.amp.autocast('cuda', enabled=True):
    #                             outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
    #
    #                             mol_cls_embedding = outputs[:, 0, :]
    #                             padded_genome_embeddings = learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
    #                             genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1).to(device)
    #                             mol_cls_embedding_genome = co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
    #                             # 把 learnable embedding 的 batch 纬 expand 作为 genome embedding 的替换
    #                             # mol_cls_embedding_genome = learnable_embedding_weight.expand(mol_cls_embedding.shape[0], -1)
    #                             mol_cls_embedding_text = co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
    #                             mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
    #                             logits = reg_head(mol_cls_embedding)
    #                             loss = criterion(logits.squeeze(), labels.squeeze())
    #
    #                         test_batch_losses.append(loss.item())
    #                         t_test_batch_losses.append(loss.item())
    #
    #                         test_batch_labels = labels.detach().cpu().flatten().tolist()
    #                         test_batch_preds = logits.detach().cpu().flatten().tolist()
    #
    #                         test_all_labels.extend(test_batch_labels)
    #                         test_all_preds.extend(test_batch_preds)
    #                         t_test_all_labels.extend(test_batch_labels)
    #                         t_test_all_preds.extend(test_batch_preds)
    #
    #                         for strain_name, label, pred in zip(strain_names, test_batch_labels, test_batch_preds):
    #                             _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
    #                             if _speceis_name is None:
    #                                 _speceis_name = strain_name_to_original_species_names_map[strain_name]
    #                             if _speceis_name not in species_wise_test_preds_dict.keys():
    #                                 species_wise_test_preds_dict[_speceis_name] = [pred]
    #                                 species_wise_test_labels_dict[_speceis_name] = [label]
    #                             else:
    #                                 species_wise_test_preds_dict[_speceis_name].append(pred)
    #                                 species_wise_test_labels_dict[_speceis_name].append(label)
    #
    #                 print('\n Calculating metrics...')
    #                 r2_test = calculate_r2(test_all_labels, test_all_preds)
    #                 spearman_test = spearmanr(test_all_labels, test_all_preds)[0]
    #                 pearson_test = pearsonr(test_all_labels, test_all_preds)[0]
    #                 gt_r2_test = calculate_r2(gt_test_all_labels, gt_test_all_preds) if len(gt_test_all_labels) > 1 else -1000
    #                 gt_spearman_test = spearmanr(gt_test_all_labels, gt_test_all_preds)[0] if len(gt_test_all_labels) > 1 else -1000
    #                 gt_pearson_test = pearsonr(gt_test_all_labels, gt_test_all_preds)[0] if len(gt_test_all_labels) > 1 else -1000
    #                 t_r2_test = calculate_r2(t_test_all_labels, t_test_all_preds) if len(t_test_all_labels) > 1 else -1000
    #                 t_spearman_test = spearmanr(t_test_all_labels, t_test_all_preds)[0] if len(t_test_all_labels) > 1 else -1000
    #                 t_pearson_test = pearsonr(t_test_all_labels, t_test_all_preds)[0] if len(t_test_all_labels) > 1 else -1000
    #
    #                 r2_MSE_spearman_pearson_species_wise = {}
    #                 for _speceis_name in species_wise_test_preds_dict.keys():
    #                     r2_species = calculate_r2(species_wise_test_labels_dict[_speceis_name], species_wise_test_preds_dict[_speceis_name])
    #                     MSE_specise = np.mean((np.array(species_wise_test_labels_dict[_speceis_name]) - np.array(species_wise_test_preds_dict[_speceis_name])) ** 2)
    #                     if len(species_wise_test_labels_dict[_speceis_name]) > 1:
    #                         spearman_species = spearmanr(species_wise_test_labels_dict[_speceis_name], species_wise_test_preds_dict[_speceis_name])[0]
    #                         pearson_species = pearsonr(species_wise_test_labels_dict[_speceis_name], species_wise_test_preds_dict[_speceis_name])[0]
    #                     else:
    #                         spearman_species = pearson_species = None
    #                     r2_MSE_spearman_pearson_species_wise[_speceis_name] = [r2_species, MSE_specise, spearman_species, pearson_species]
    #
    #                 if r2_test > best_R2_test:
    #                     best_R2_test = r2_test
    #                     best_test_prdictions = test_all_preds
    #
    #                     torch.save({
    #                         'R2': best_R2_test,
    #                         'optimizer_state_dict': optimizer.state_dict(),
    #                         'ChemBERTa_state_dict': ChemBERTa_model.state_dict(),
    #                         're_head_state_dict': reg_head.state_dict(),
    #                         'cls_head_state_dict': cls_head.state_dict(),
    #                         'co_cross_attn_genome': co_cross_attn_genome.state_dict(),
    #                         'co_cross_attn_text': co_cross_attn_text.state_dict(),
    #                         'learnable_embedding_weight': learnable_embedding_weight
    #                     }, model_save_dir / f'genome_text_learnable_emb_Strain_wise_best_R2_group_{args.test_group}_ensemble_{ensemble}.pth')
    #
    #                 if spearman_test > best_spearman_test:
    #                     best_spearman_test = spearman_test
    #
    #                 if pearson_test > best_pearson_test:
    #                     best_pearson_test = pearson_test
    #
    #                 logger.info(f'\n Test species wise R2, MSE, Spearman, Pearson:')
    #                 for species_name, metrics in r2_MSE_spearman_pearson_species_wise.items():
    #                     formatted_metrics = ", ".join(f"{m:.4f}" if isinstance(m, float) else str(m) for m in metrics)
    #                     logger.info(f'    {species_name}:  {formatted_metrics}')
    #
    # #                 print(f""" Ensemble {ensemble+1}/{num_ensembles} Epoch {epoch + 1}/{num_epochs}
    # # Training Loss: {np.array(train_batch_losses).mean():.4f}, Test Loss: {np.array(test_batch_losses).mean():.4f}
    # #   Genome text Training Loss: {np.array(gt_train_batch_losses).mean():.4f}, Genome Text Test Loss: {np.array(gt_test_batch_losses).mean():.4f}
    # #   Text only Training Loss: {np.array(t_train_batch_losses).mean():.4f}, Text only Test Loss: {np.array(t_test_batch_losses).mean():.4f}
    # # Train R2: {r2_train:.4f}, Test R2: {r2_test:.4f}, Best Test R2: {best_R2_test:.4f}
    # #   Genome Text Train R2: {gt_r2_train:.4f}, Genome Text Test R2: {gt_r2_test:.4f}
    # #   Text only Train R2: {t_r2_train:.4f}, Text only Test R2: {t_r2_test:.4f}
    # # Train spearman:{spearman_train:.4f}, Test spearman:{spearman_test:.4f}, Best Test spearman:{best_spearman_test:.4f}
    # #   Genome Text Train spearman:{gt_spearman_train:.4f}, Genome Text Test spearman:{gt_spearman_test:.4f}
    # #   Text only Train spearman:{t_spearman_train:.4f}, Text only Test spearman:{t_spearman_test:.4f}
    # # Train pearson:{pearson_train:.4f}, Test pearson:{pearson_test:.4f}, Best Test pearson:{best_pearson_test:.4f}
    # #   Genome Text Train pearson:{gt_pearson_train:.4f}, Genome Text Test pearson:{gt_pearson_test:.4f}
    # #   Text onlyTrain pearson:{t_pearson_train:.4f}, Text only Test pearson:{t_pearson_test:.4f}""")
                logger.info(f"""\n Ensemble {ensemble + 1}/{num_ensembles} Epoch {epoch + 1}/{num_epochs}
    Regression Training Loss: {np.array(train_batch_losses).mean():.4f}
      Genome text Training Loss: {np.array(gt_train_batch_losses).mean():.4f}
      Text only Training Loss: {np.array(t_train_batch_losses).mean():.4f}
    Train R2: {r2_train:.4f}
      Genome Text Train R2: {gt_r2_train:.4f}
      Text only Train R2: {t_r2_train:.4f}
    Train spearman:{spearman_train:.4f}
      Genome Text Train spearman:{gt_spearman_train:.4f}
      Text only Train spearman:{t_spearman_train:.4f}
    Train pearson:{pearson_train:.4f}
      Genome Text Train pearson:{gt_pearson_train:.4f}
      Text onlyTrain pearson:{t_pearson_train:.4f}""")
                    # logger.info(f"""\n Ensemble {ensemble + 1}/{num_ensembles} Epoch {epoch + 1}/{num_epochs}
                    # Regression Training Loss: {np.array(train_batch_losses).mean():.4f}, Test Loss: {np.array(test_batch_losses).mean():.4f}
                    #   Genome text Training Loss: {np.array(gt_train_batch_losses).mean():.4f}, Genome Text Test Loss: {np.array(gt_test_batch_losses).mean():.4f}
                    #   Text only Training Loss: {np.array(t_train_batch_losses).mean():.4f}, Text only Test Loss: {np.array(t_test_batch_losses).mean():.4f}
                    # Classification Training Loss: {np.array(cls_train_batch_losses).mean():.4f}
                    #   Genome text Training Loss: {np.array(cls_gt_train_batch_losses).mean():.4f}
                    #   Text only Training Loss: {np.array(cls_t_train_batch_losses).mean():.4f}
                    # Train R2: {r2_train:.4f}, Test R2: {r2_test:.4f}, Best Test R2: {best_R2_test:.4f}
                    #   Genome Text Train R2: {gt_r2_train:.4f}, Genome Text Test R2: {gt_r2_test:.4f}
                    #   Text only Train R2: {t_r2_train:.4f}, Text only Test R2: {t_r2_test:.4f}
                    # Train spearman:{spearman_train:.4f}, Test spearman:{spearman_test:.4f}, Best Test spearman:{best_spearman_test:.4f}
                    #   Genome Text Train spearman:{gt_spearman_train:.4f}, Genome Text Test spearman:{gt_spearman_test:.4f}
                    #   Text only Train spearman:{t_spearman_train:.4f}, Text only Test spearman:{t_spearman_test:.4f}
                    # Train pearson:{pearson_train:.4f}, Test pearson:{pearson_test:.4f}, Best Test pearson:{best_pearson_test:.4f}
                    #   Genome Text Train pearson:{gt_pearson_train:.4f}, Genome Text Test pearson:{gt_pearson_test:.4f}
                    #   Text onlyTrain pearson:{t_pearson_train:.4f}, Text only Test pearson:{t_pearson_test:.4f}""")

                if r2_train > best_R2_test:
                    best_R2_test = r2_train
                    torch.save({
                        'R2': best_R2_test,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'mdlm_model_state_dict': mdlm_model.state_dict(),
                        're_head_state_dict': reg_head.state_dict(),
                        'cls_head_state_dict': cls_head.state_dict(),
                        'co_cross_attn_genome': co_cross_attn_genome.state_dict(),
                        'co_cross_attn_text': co_cross_attn_text.state_dict(),
                        'learnable_embedding_weight': learnable_embedding_weight
                    },
                        model_save_dir / f'noise_guidance_best_R2_all_peptide_epoch_{num_epochs}.pth')

                if (epoch + 1) % 10 ==0:
                    # best_R2_test = r2_train
                    torch.save({
                        'R2': best_R2_test,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'mdlm_model_state_dict': mdlm_model.state_dict(),
                        're_head_state_dict': reg_head.state_dict(),
                        'cls_head_state_dict': cls_head.state_dict(),
                        'co_cross_attn_genome': co_cross_attn_genome.state_dict(),
                        'co_cross_attn_text': co_cross_attn_text.state_dict(),
                        'learnable_embedding_weight': learnable_embedding_weight
                    },
                        model_save_dir / f'noise_guidance_all_peptide_epoch_{epoch + 1}_of_{num_epochs}.pth')

        # if best_test_prdictions is not None:
        #     if best_R2_test > -10:
        #         test_predictions_of_ensembles.append(best_test_prdictions)
    # print(f'\n len of ensembled test predictions: {len(test_predictions_of_ensembles)}')
    # logger.info(f'\n len of ensembled test predictions: {len(test_predictions_of_ensembles)}')
    # test_predictions_of_ensembles = np.array(test_predictions_of_ensembles)
    # ensembled_predictions = np.mean(test_predictions_of_ensembles, axis=0)
    # ensembled_R2 = calculate_r2(test_all_labels, ensembled_predictions)
    # ensembled_spearman = spearmanr(test_all_labels, ensembled_predictions)[0]
    # ensembled_pearson = pearsonr(test_all_labels, ensembled_predictions)[0]
    #
    # # print(f'\n Ensemble R2 of {args.test_group}: {ensembled_R2:.4f}')
    # # print(f' Ensemble spearman of {args.test_group}: {ensembled_spearman:.4f}')
    # # print(f' Ensemble pearson of {args.test_group}: {ensembled_pearson:.4f}')
    # logger.info(f'\n Ensemble R2 of {args.test_group}: {ensembled_R2:.4f}')
    # logger.info(f' Ensemble spearman of {args.test_group}: {ensembled_spearman:.4f}')
    # logger.info(f' Ensemble pearson of {args.test_group}: {ensembled_pearson:.4f}')