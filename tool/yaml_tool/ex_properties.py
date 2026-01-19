import re
import yaml


def properties_to_yaml(properties_content):
    """
    将properties格式内容转换为YAML格式
    """
    result = {}

    # 按行分割内容
    lines = properties_content.strip().split('\n')

    for line in lines:
        # 跳过空行和注释行
        if not line.strip() or line.strip().startswith('#'):
            continue

        # 分割键值对
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # 处理值中的转义字符
            value = value.replace('\\\n', '')  # 移除行继续符

            # 解析嵌套结构（JSON格式的值）
            if value.startswith('{') and value.endswith('}'):
                try:
                    # 简单的JSON解析（处理嵌套字典）
                    value_dict = {}
                    # 移除外层花括号并按逗号分割
                    inner_content = value[1:-1].strip()
                    if inner_content:
                        # 处理嵌套的键值对
                        pairs = re.split(r',(?![^{]*\})', inner_content)
                        for pair in pairs:
                            if ':' in pair:
                                k, v = pair.split(':', 1)
                                k = k.strip().strip('"\'')
                                v = v.strip().strip('"\'')
                                value_dict[k] = v
                    value = value_dict
                except:
                    # 如果解析失败，保持原样
                    pass

            # 处理逗号分隔的列表值
            elif ',' in value and not value.startswith('"'):
                value = [item.strip() for item in value.split(',')]

            # 构建嵌套的YAML结构
            keys = key.split('.')
            current_level = result

            # 遍历嵌套键
            for i, k in enumerate(keys):
                if i == len(keys) - 1:
                    # 最后一个键，设置值
                    current_level[k] = value
                else:
                    # 中间键，确保字典存在
                    if k not in current_level:
                        current_level[k] = {}
                    current_level = current_level[k]

    return result


def convert_properties_to_yml(input_file, output_file):
    """
    将properties文件转换为YAML文件
    """
    # 读取properties文件
    with open(input_file, 'r', encoding='utf-8') as f:
        properties_content = f.read()

    # 转换为YAML格式
    yaml_data = properties_to_yaml(properties_content)

    # 写入YAML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, indent=2)

    print(f"转换完成！YAML文件已保存为: {output_file}")


# 如果直接运行此脚本，使用示例
if __name__ == "__main__":
    # 示例：直接处理提供的文本内容
    properties_text = """# 定义资源字段名。key-value形式，value以逗号分隔。
i1.5G.resource.fieldNamesDef = {"CGNB":"gNB_id,userlabel,gNB_model,gNBIdLength,software_version,freq_mode,cel_num,serialid,clocksource,\\
  synchronous_mode,bbu_longitude,bbu_latitude,gNB_type,gNB_Deployment_mode,masterOperator_PLMN,physical_userlabel",\\
  "CCCU":"master_PHY_Cell_id,related_gNB_dn,related_gNB_id,related_gNB_userlabel,cel_id,celCU_id_local,userlabel", \\
  "CCDU":"master_PHY_Cell_id,related_masterOperator_gNB_id,related_masterOperator_gNB_userlabel,userlabel,celDU_id_local,PCI,freq_ul,freq_dl,freq_pointno_ul\\
,freq_pointno_dl,bandwidth_ul,bandwidth_dl,TDD_first_TransmissionPeriodicity,TDD_first_TransmissionPeriodicity_type,TDD_first_nrofDownlinkSymbols\\
,TDD_first_nrofUplinkSymbols,S_slot_GPSymbols,TDD_second_id,TDD_second_TransmissionPeriodicity,TDD_second_TransmissionPeriodicity_type,TDD_second_nrofDownlinkSymbols\\
,TDD_second_nrofUplinkSymbols",\\
  "CLGNB":"gNB_id,userlabel,gNBIdLength,ip_related_CP,Ng_bandwidth,master_PHY_gNB_id",\\
  "CLCCU":"master_PHY_Cell_id,related_gNB_dn,related_gNB_id,related_gNB_userlabel,cel_id,celCU_id_local,userlabel",\\
   "CLCDU":"master_PHY_Cell_id,related_gNB_dn,related_gNB_id,related_gNB_userlabel,cel_id,celDU_id_local,userlabel,TAC",\\
    "CLCCUS":"slice_InCellCU_ID,related_CelCU_dn,related_Cel_id,userlabel,PLMN,sst,sd",\\
     "CLCDUS":"slice_InCellDU_ID,related_CellDU_dn,related_CelDU_id,userlabel,PLMN,sst,sd,vlan_id_AMF,vlan_id_UPF,tac",\\
      "CRRU":"related_masterOperator_gNB_dn,related_masterOperator_gNB_id,related_masterOperator_gNB_userlabel,rru_vendor_location,userlabel,freq_mode,rru_model,\\
     cel_dn_list,cel_id_list,serialid,MaxActPower_rru,rru_type,related_PHUB_dn,related_PHUB_id,related_PHUB_userlabel,rru_longitude,rru_latitude,rru_id",\\
      "CUNIT":"related_masterOperator_gNB_dn,related_masterOperator_gNB_id,related_masterOperator_gNB_userlabel,upk_type,upk_location,serialid,DateOfManufacture,DateOfLastService",\\
      "CPHUB":"related_masterOperator_gNB_dn,related_masterOperator_gNB_id,related_masterOperator_gNB_userlabel,PHUB_vendor_location,PHUB_id,\\
         userlabel,PHUB_model,PHUB_port,PHUB_cascadingmode,serialid,software_version",\\
      "COMCR":"omc_dn,omc_id,userlabel,omc_model,omc_ip,software_version,software_version_FM,software_version_CM,software_version_PM",\\
      "CLCON":"ID,license_userlabel,license_class,related_dn,allocated_value,used_value,expire_date,description",\\
      "LGNBP":"gNB_id,gNB_userlabel,encrypAlgPriority,integProtAlgPriority,RlcMode,PDCPConfig",\\
      "LCDUP":"master_PHY_Cell_id,related_gNB_dn,related_gNB_id,related_gNB_userlabel,cel_id,celDU_id_local,userlabel,T304,logicalChannelPara,MACConfig,RLCConfig,CDRX,",\\
      "LCCUP":"master_PHY_Cell_id,related_gNB_dn,related_gNB_id,related_gNB_userlabel,cel_id,celCU_id_local,userlabel,PlmnIdList,threshServingLowP,cellReselectionPriority,sNonIntraSearchP,Qhyst,\\
      IntraTReselectionNR,IntraQrxLevMin,sIntraSearchP,InterFreqCarrierFreq,a3_IntraFreq_Coverage,a1_InterFreq_Coverage,a2_InterFreq_Coverage,a5_InterFreq_Coverage,a1_InterRAT_Coverage,a2_InterRAT_Coverage,\\
      b1_InterRAT_Coverage,b2_InterRAT_Coverage,PscellA2RsrpThld,RlcMode,NsaDcSwitch,SIB2Config,intraFreqNeighCellInfo,tReselectionEUTRA,T320,ranPagingCycle,PDCPConfig,RLCConfig,ExternalNRFreq,externalNRCell,\\
      NRCellRelation,EUTRAFreq,dlpdcpsplitmode,ulprimarypath,a2Thresholdnsa,hysteresisa2nsa,timeToTriggera2nsa,VoNRSwitch,secratusagereportswitch,CDRX,cel_Deployment_mode,ca_cel_tag,ref_ca_cel",\\
      "CDUP":"master_PHY_Cell_id,related_gNB_dn,related_masterOperator_gNB_id,related_masterOperator_gNB_userlabel,userlabel,QRxLevMin,Qrxlevminoffset,SCSSpecificCarrierListDL,SCSSpecificCarrierListUL,\\
      PuschPcSwitch,PucchPcSwitch,UlMuMimoSwitch,DlMuMimoSwitch,defaultPagingCycle,nAndPagingFrameOffset,ns,ssbPeriodicityServingCell,ssPBCHBlockPower,configuredMaxTxPower,bwpList,rachOccasionNumber,\\
      powerRampingStep,preambleReceivedTargetPower,preambleTransMax,raResponseWindow,prachConfigurationIndex,ssb-perRACH-OccasionAndCB-PreamblesPerSSB,msg1SubCarrierSpacing,restrictedSetConfig,msg3TransformPrecoder,\\
      rsrpThresholdSSB,prach-RootSequenceIndex,P0NominalPUCCH,P0NominalwithgrantPUSCH,deltaPreambleMsg3,SsbSubcarrierSpacing,FrequencyBand,AdditionalFrequencyBand,T300,T301,T310,T311,N310,N311,ssbFrequency,MIBConfig,\\
      PLMNConfig,SIBSchedule,PagingConfig,PrachConfig,cellSelectionInfoConfig,PUCCHConfig,PUSCHConfig,PDSCHConfig,PDCCHConfig,SsPBCHBlockPower,p0NominalWithoutGrantPUSCH,alphaPUSCH,alphaMsg3,T319",\\
      "CANT":"related_masterOperator_gNB_dn,related_ret_rru_dn,related_rru_dn,related_cel_dn,related_masterOperator_gNB_id,related_masterOperator_gNB_userlabel,ret_rru_code,rcu_code,retsubunit,vendor_name,tilt,rcu_serialid",\\
      "CBEAM":"master_PHY_Cell_id,master_PHY_Cell_DN,beamIndex,azimuth,tilt,beamWidthHorizontal,beamWidthVertical",\\
      "COLOG":"log_id,StartTime,operating_Account,operating_IP,operation_Type,operation_Name,objectOfOperation,operationContent,result",\\
      "CFHL":"related_masterOperator_gNB_dn,related_masterOperator_gNB_id,related_masterOperator_gNB_userlabel,frontHaulLink_id,relatedUp_opticalModule_id,relatedDown_opticalModule_id",\\
      "COPM":"related_masterOperator_gNB_dn,related_masterOperator_gNB_id,related_masterOperator_gNB_userlabel,related_rru_dn,related_rru_id,related_ rru_userlabel,opticalModule_type,opticalModule_capacity,opticalModule_id,\\
       opticalModule_vendor_location,serialid,vendor_name,opticalModule_model,protocolType,dateOfManufacture,dateOfLastService"}
#设备类型。以逗号分隔。
i1.5G.file.type=CGNB,CCCU,CCDU,CLGNB,CLCCU,CLCDU,CLCCUS,CLCDUS,CRRU,CBEAM,CUNIT,COPM,CFHL,CPHUB,COMCR,CANT,CLCON,COLOG,LGNBP,CDUP,LCCUP,LCDUP"""

    # 转换为YAML
    yaml_data = properties_to_yaml(properties_text)

    # 输出YAML格式
    yaml_output = yaml.dump(yaml_data, allow_unicode=True, default_flow_style=False, indent=2)
    print("转换后的YAML格式:")
    print(yaml_output)

    # 保存到文件（可选）
    with open('output.yml', 'w', encoding='utf-8') as f:
        f.write(yaml_output)

# 使用方法：
# 1. 将properties文件保存为 input.properties
# 2. 调用 convert_properties_to_yml('input.properties', 'output.yml')