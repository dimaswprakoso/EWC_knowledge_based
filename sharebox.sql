/*
Navicat MySQL Data Transfer

Source Server         : localhost_3306
Source Server Version : 50717
Source Host           : localhost:3306
Source Database       : sharebox

Target Server Type    : MYSQL
Target Server Version : 50717
File Encoding         : 65001

Date: 2019-04-23 11:22:26
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for ewc_level1
-- ----------------------------
DROP TABLE IF EXISTS `ewc_level1`;
CREATE TABLE `ewc_level1` (
  `id` varchar(255) DEFAULT NULL,
  `EWC_level1` varchar(255) DEFAULT NULL,
  `EWC_id` varchar(255) DEFAULT NULL,
  `description` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of ewc_level1
-- ----------------------------
INSERT INTO `ewc_level1` VALUES ('1', '1', '1', 'Wastes resulting from exploration, mining, quarrying, physical and chemical\r treatment of minerals');
INSERT INTO `ewc_level1` VALUES ('2', '2', '2', 'Wastes from agriculture, horticulture, aquaculture, forestry, hunting and fishing \rfood preparation and processing');
INSERT INTO `ewc_level1` VALUES ('3', '3', '3', 'Wastes from wood processing and the production of panels and furniture, pulp,\r paper and cardboard');
INSERT INTO `ewc_level1` VALUES ('4', '4', '4', 'Wastes from the leather, fur and textile industries');
INSERT INTO `ewc_level1` VALUES ('5', '5', '5', 'Wastes from petroleum refining, natural gas purification and pyrolytic treatment of coal');
INSERT INTO `ewc_level1` VALUES ('6', '6', '6', 'Wastes from inorganic chemical processes');
INSERT INTO `ewc_level1` VALUES ('7', '7', '7', 'Wastes from organic chemical processes');
INSERT INTO `ewc_level1` VALUES ('8', '8', '8', 'Wastes from the manufacture, formulation, supply and use (MFSU) of coatings\r (paints, varnishes and vitreous enamels), adhesives, sealants and printing inks');
INSERT INTO `ewc_level1` VALUES ('9', '9', '9', 'Wastes from the photographicindustry');
INSERT INTO `ewc_level1` VALUES ('10', '10', '10', 'Wastes from thermal processes');
INSERT INTO `ewc_level1` VALUES ('11', '11', '11', 'Wastes from chemical surface treatment and coating of metals and other\r materials; non-ferrous hydro-metallurgy');
INSERT INTO `ewc_level1` VALUES ('12', '12', '12', 'Wastes from shaping and physical and mechanical surface treatment of metals\r and plastics');
INSERT INTO `ewc_level1` VALUES ('13', '13', '13', 'Oil wastes and wastes of liquid fuels (except edible oils, 05 and 12)');
INSERT INTO `ewc_level1` VALUES ('14', '14', '14', 'Waste organic solvents, refrigerants and propellants (except 07 and 08)');
INSERT INTO `ewc_level1` VALUES ('15', '15', '15', 'Waste packaging; absorbents, wiping cloths, filter materials and protective\r clothing not otherwise specified');
INSERT INTO `ewc_level1` VALUES ('16', '16', '16', 'Wastes not otherwise specified in the list');
INSERT INTO `ewc_level1` VALUES ('17', '17', '17', 'Construction and demolition wastes (including excavated soil from contaminated\r\nsites)');
INSERT INTO `ewc_level1` VALUES ('18', '18', '18', 'Wastes from human or animal health care and/or related research (except\r kitchen and restaurant wastes not arising from immediate health care)');
INSERT INTO `ewc_level1` VALUES ('19', '19', '19', 'Wastes from waste management facilities, off-site waste water treatment plants and the preparation of water intended for human consumption and water for industrial use');
INSERT INTO `ewc_level1` VALUES ('20', '20', '20', 'Municipal wastes (household waste and similar commercial, industrial and institutional wastes) including separately collected fractions');

-- ----------------------------
-- Table structure for ewc_level2
-- ----------------------------
DROP TABLE IF EXISTS `ewc_level2`;
CREATE TABLE `ewc_level2` (
  `id` varchar(255) DEFAULT NULL,
  `parent_id` varchar(255) DEFAULT NULL,
  `EWC_level2` varchar(255) DEFAULT NULL,
  `EWC_id` varchar(255) DEFAULT NULL,
  `description` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of ewc_level2
-- ----------------------------
INSERT INTO `ewc_level2` VALUES ('1', '1', '1', '01 01', 'wastes from mineral excavation');
INSERT INTO `ewc_level2` VALUES ('2', '1', '3', '01 03', 'wastes from physical and chemical processing of metalliferous minerals');
INSERT INTO `ewc_level2` VALUES ('3', '1', '4', '01 04', 'wastes from physical and chemical processing of non-metalliferous minerals');
INSERT INTO `ewc_level2` VALUES ('4', '1', '5', '01 05', 'drilling muds and other drilling wastes');
INSERT INTO `ewc_level2` VALUES ('5', '2', '1', '02 01', 'wastes from agriculture, horticulture, aquaculture, forestry, hunting and fishing');
INSERT INTO `ewc_level2` VALUES ('6', '2', '2', '02 02', 'wastes from the preparation and processing of meat, fish and other foods of animal origin');
INSERT INTO `ewc_level2` VALUES ('7', '2', '3', '02 03', 'wastes from fruit, vegetables, cereals, edible oils, cocoa, coffee, tea and tobacco preparation and processing; conserve production; yeast and yeast extract production, molasses preparation and fermentation');
INSERT INTO `ewc_level2` VALUES ('8', '2', '4', '02 04', 'wastes from sugar processing');
INSERT INTO `ewc_level2` VALUES ('9', '2', '5', '02 05', 'wastes from the dairy products industry');
INSERT INTO `ewc_level2` VALUES ('10', '2', '6', '02 06', 'wastes from the baking and confectionery industry');
INSERT INTO `ewc_level2` VALUES ('11', '2', '7', '02 07', 'wastes from the production of alcoholic and non-alcoholic beverages (except coffee, tea and cocoa)');
INSERT INTO `ewc_level2` VALUES ('12', '3', '1', '03 01', 'wastes from wood processing and the production of panels and furniture');
INSERT INTO `ewc_level2` VALUES ('13', '3', '2', '03 02', 'wastes from wood preservation');
INSERT INTO `ewc_level2` VALUES ('14', '3', '3', '03 03', 'wastes from pulp, paper and cardboard production and processing');
INSERT INTO `ewc_level2` VALUES ('15', '4', '1', '04 01', 'wastes from the leather and fur industry');
INSERT INTO `ewc_level2` VALUES ('16', '4', '2', '04 02', 'wastes from the textile industry');
INSERT INTO `ewc_level2` VALUES ('17', '5', '1', '05 01', 'wastes from petroleum refining');
INSERT INTO `ewc_level2` VALUES ('18', '5', '6', '05 06', 'wastes from the pyrolytic treatment of coal');
INSERT INTO `ewc_level2` VALUES ('19', '5', '7', '05 07', 'wastes from natural gas purification and transportation');
INSERT INTO `ewc_level2` VALUES ('20', '6', '1', '06 01', 'wastes from the manufacture, formulation, supply and use (MFSU) of acids');
INSERT INTO `ewc_level2` VALUES ('21', '6', '2', '06 02', 'wastes from the MFSU of bases');
INSERT INTO `ewc_level2` VALUES ('22', '6', '3', '06 03', 'wastes from the MFSU of salts and their solutions and metallic oxides');
INSERT INTO `ewc_level2` VALUES ('23', '6', '4', '06 04', 'metal-containing wastes other than those mentioned in 06 03');
INSERT INTO `ewc_level2` VALUES ('24', '6', '5', '06 05', 'Sludges from on-site effluent treatment');
INSERT INTO `ewc_level2` VALUES ('25', '6', '6', '06 06', 'wastes from the MFSU of sulphur chemicals, sulphur chemical processes and desulphurisation processes');
INSERT INTO `ewc_level2` VALUES ('26', '6', '7', '06 07', 'wastes from the MFSU of halogens and halogen chemical processes');
INSERT INTO `ewc_level2` VALUES ('27', '6', '8', '06 08', 'wastes from the MFSU of silicon and silicon derivatives');
INSERT INTO `ewc_level2` VALUES ('28', '6', '9', '06 09', 'wastes from the MSFU of phosphorous chemicals and phosphorous chemical processes');
INSERT INTO `ewc_level2` VALUES ('29', '6', '10', '06 10', 'wastes from the MFSU of nitrogen chemicals, nitrogen chemical processes and fertiliser manufacture');
INSERT INTO `ewc_level2` VALUES ('30', '6', '11', '06 11', 'wastes from the manufacture of inorganic pigments and opacificiers');
INSERT INTO `ewc_level2` VALUES ('31', '6', '13', '06 13', 'wastes from inorganic chemical processes not otherwise specified');
INSERT INTO `ewc_level2` VALUES ('32', '7', '1', '07 01', 'wastes from the manufacture, formulation, supply and use (MFSU) of basic organic chemicals');
INSERT INTO `ewc_level2` VALUES ('33', '7', '2', '07 02', 'wastes from the MFSU of plastics, synthetic rubber and man-made fibres');
INSERT INTO `ewc_level2` VALUES ('34', '7', '3', '07 03', 'wastes from the MFSU of organic dyes and pigments (except 06 11)');
INSERT INTO `ewc_level2` VALUES ('35', '7', '4', '07 04', 'wastes from the MFSU of organic plant protection products (except 02 01 08 and 02 01 09), wood preserving agents (except 03 02) and other biocides');
INSERT INTO `ewc_level2` VALUES ('36', '7', '5', '07 05', 'wastes from the MFSU of pharmaceuticals');
INSERT INTO `ewc_level2` VALUES ('37', '7', '6', '07 06', 'wastes from the MFSU of fats, grease, soaps, detergents, disinfectants and cosmetics');
INSERT INTO `ewc_level2` VALUES ('38', '7', '7', '07 07', 'wastes from the MFSU of fine chemicals and chemical products not otherwise specified');
INSERT INTO `ewc_level2` VALUES ('39', '8', '1', '08 01', 'wastes from MFSU and removal of paint and varnish');
INSERT INTO `ewc_level2` VALUES ('40', '8', '2', '08 02', 'wastes from MFSU of other coatings (including ceramic materials)');
INSERT INTO `ewc_level2` VALUES ('41', '8', '3', '08 03', 'wastes from MFSU of printing inks');
INSERT INTO `ewc_level2` VALUES ('42', '8', '4', '08 04', 'wastes from MFSU of adhesives and sealants (including waterproofing products)');
INSERT INTO `ewc_level2` VALUES ('43', '8', '5', '08 05', 'wastes not otherwise specified in 08');
INSERT INTO `ewc_level2` VALUES ('44', '9', '1', '09 01', 'wastes from the photographic industry');
INSERT INTO `ewc_level2` VALUES ('45', '10', '1', '10 01', 'wastes from power stations and other combustion plants (except 19)');
INSERT INTO `ewc_level2` VALUES ('46', '10', '2', '10 02', 'wastes from the iron and steel industry');
INSERT INTO `ewc_level2` VALUES ('47', '10', '3', '10 03', 'wastes from aluminium thermal metallurgy');
INSERT INTO `ewc_level2` VALUES ('48', '10', '4', '10 04', 'wastes from lead thermal metallurgy');
INSERT INTO `ewc_level2` VALUES ('49', '10', '5', '10 05', 'wastes from zinc thermal metallurgy');
INSERT INTO `ewc_level2` VALUES ('50', '10', '6', '10 06', 'wastes from copper thermal metallurgy');
INSERT INTO `ewc_level2` VALUES ('51', '10', '7', '10 07', 'wastes from silver, gold and platinum thermal metallurgy');
INSERT INTO `ewc_level2` VALUES ('52', '10', '8', '10 08', 'wastes from other non-ferrous thermal metallurgy');
INSERT INTO `ewc_level2` VALUES ('53', '10', '9', '10 09', 'wastes from casting of ferrous pieces');
INSERT INTO `ewc_level2` VALUES ('54', '10', '10', '10 10', 'wastes from casting of non-ferrous pieces');
INSERT INTO `ewc_level2` VALUES ('55', '10', '11', '10 11', 'wastes from manufacture of glass and glass products');
INSERT INTO `ewc_level2` VALUES ('56', '10', '12', '10 12', 'wastes from manufacture of ceramic goods, bricks, tiles and construction products');
INSERT INTO `ewc_level2` VALUES ('57', '10', '13', '10 13', 'wastes from manufacture of cement, lime and plaster and articles and products made from them');
INSERT INTO `ewc_level2` VALUES ('58', '10', '14', '10 14', 'waste from crematoria');
INSERT INTO `ewc_level2` VALUES ('59', '11', '1', '11 01', 'wastes from chemical surface treatment and coating of metals and other materials (for example galvanic processes, zinc coating processes, pickling processes, etching, phosphating, alkaline degreasing, anodising)');
INSERT INTO `ewc_level2` VALUES ('60', '11', '2', '11 02', 'wastes from non-ferrous hydrometallurgical processes');
INSERT INTO `ewc_level2` VALUES ('61', '11', '3', '11 03', 'sludges and solids from tempering processes');
INSERT INTO `ewc_level2` VALUES ('62', '11', '5', '11 05', 'wastes from hot galvanising processes');
INSERT INTO `ewc_level2` VALUES ('63', '12', '1', '12 01', 'wastes from shaping and physical and mechanical surface treatment of metals and plastics');
INSERT INTO `ewc_level2` VALUES ('64', '12', '3', '12 03', 'wastes from water and steam degreasing processes (except 11)');
INSERT INTO `ewc_level2` VALUES ('65', '13', '1', '13 01', 'waste hydraulic oils');
INSERT INTO `ewc_level2` VALUES ('66', '13', '2', '13 02', 'waste engine, gear and lubricating oils');
INSERT INTO `ewc_level2` VALUES ('67', '13', '3', '13 03', 'waste insulating and heat transmission oils');
INSERT INTO `ewc_level2` VALUES ('68', '13', '4', '13 04', 'bilge oils');
INSERT INTO `ewc_level2` VALUES ('69', '13', '5', '13 05', 'oil/water separator contents');
INSERT INTO `ewc_level2` VALUES ('70', '13', '7', '13 07', 'wastes of liquid fuels');
INSERT INTO `ewc_level2` VALUES ('71', '13', '8', '13 08', 'oil wastes not otherwise specified');
INSERT INTO `ewc_level2` VALUES ('72', '14', '6', '14 06', 'waste organic solvents, refrigerants and foam/aerosol propellants');
INSERT INTO `ewc_level2` VALUES ('73', '15', '1', '15 01', 'packaging (including separately collected municipal packaging waste)');
INSERT INTO `ewc_level2` VALUES ('74', '15', '2', '15 02', 'absorbents, filter materials, wiping cloths and protective clothing');
INSERT INTO `ewc_level2` VALUES ('75', '16', '1', '16 01', 'end-of-life vehicles from different means of transport (including off-road machinery) and wastes from dismantling of end-of-life vehicles and vehicle maintenance (except 13, 14, 16 06 and 16 08)');
INSERT INTO `ewc_level2` VALUES ('76', '16', '2', '16 02', 'wastes from electrical and electronic equipment');
INSERT INTO `ewc_level2` VALUES ('77', '16', '3', '16 03', 'off-specification batches and unused products');
INSERT INTO `ewc_level2` VALUES ('78', '16', '4', '16 04', 'waste explosives');
INSERT INTO `ewc_level2` VALUES ('79', '16', '5', '16 05', 'gases in pressure containers and discarded chemicals');
INSERT INTO `ewc_level2` VALUES ('80', '16', '6', '16 06', 'batteries and accumulators');
INSERT INTO `ewc_level2` VALUES ('81', '16', '7', '16 07', 'wastes from transport tank, storage tank and barrel cleaning (except 05 and 13)');
INSERT INTO `ewc_level2` VALUES ('82', '16', '8', '16 08', 'spent catalysts');
INSERT INTO `ewc_level2` VALUES ('83', '16', '9', '16 09', 'oxidising substances');
INSERT INTO `ewc_level2` VALUES ('84', '16', '10', '16 10', 'aqueous liquid wastes destined for off-site treatment');
INSERT INTO `ewc_level2` VALUES ('85', '16', '11', '16 11', 'waste linings and refractories');
INSERT INTO `ewc_level2` VALUES ('86', '17', '1', '17 01', 'concrete, bricks, tiles and ceramics');
INSERT INTO `ewc_level2` VALUES ('87', '17', '2', '17 02', 'wood, glass and plastic');
INSERT INTO `ewc_level2` VALUES ('88', '17', '3', '17 03', 'bituminous mixtures, coal tar and tarred products');
INSERT INTO `ewc_level2` VALUES ('89', '17', '4', '17 04', 'metals (including their alloys)');
INSERT INTO `ewc_level2` VALUES ('90', '17', '5', '17 05', 'soil (including excavated soil from contaminated sites), stones and dredging spoil');
INSERT INTO `ewc_level2` VALUES ('91', '17', '6', '17 06', 'insulation materials and asbestos-containing construction materials');
INSERT INTO `ewc_level2` VALUES ('92', '17', '8', '17 08', 'gypsum-based construction material');
INSERT INTO `ewc_level2` VALUES ('93', '17', '9', '17 09', 'other construction and demolition wastes');
INSERT INTO `ewc_level2` VALUES ('94', '18', '1', '18 01', 'wastes from natal care, diagnosis, treatment or prevention of disease in humans');
INSERT INTO `ewc_level2` VALUES ('95', '18', '2', '18 02', 'wastes from research, diagnosis, treatment or prevention of disease involving animals');
INSERT INTO `ewc_level2` VALUES ('96', '19', '1', '19 01', 'wastes from incineration or pyrolysis of waste');
INSERT INTO `ewc_level2` VALUES ('97', '19', '2', '19 02', 'wastes from physico/chemical treatments of waste (including dechromatation, decyanidation, neutralisation)');
INSERT INTO `ewc_level2` VALUES ('98', '19', '3', '19 03', 'stabilised/solidified wastes (4)');
INSERT INTO `ewc_level2` VALUES ('99', '19', '4', '19 04', 'vitrified waste and wastes from vitrification');
INSERT INTO `ewc_level2` VALUES ('100', '19', '5', '19 05', 'wastes from aerobic treatment of solid wastes');
INSERT INTO `ewc_level2` VALUES ('101', '19', '6', '19 06', 'wastes from anaerobic treatment of waste');
INSERT INTO `ewc_level2` VALUES ('102', '19', '7', '19 07', 'landfill leachate');
INSERT INTO `ewc_level2` VALUES ('103', '19', '8', '19 08', 'wastes from waste water treatment plants not otherwise specified');
INSERT INTO `ewc_level2` VALUES ('104', '19', '9', '19 09', 'wastes from the preparation of water intended for human consumption or water for industrial use');
INSERT INTO `ewc_level2` VALUES ('105', '19', '10', '19 10', 'wastes from oil regeneration');
INSERT INTO `ewc_level2` VALUES ('106', '19', '11', '19 11', 'wastes from shredding of metal-containing wastes');
INSERT INTO `ewc_level2` VALUES ('107', '19', '12', '19 12', 'wastes from the mechanical treatment of waste (for example sorting, crushing, compacting, pelletising) not otherwise specified');
INSERT INTO `ewc_level2` VALUES ('108', '19', '13', '19 13', 'wastes from soil and groundwater remediation');
INSERT INTO `ewc_level2` VALUES ('109', '20', '1', '20 01', 'separately collected fractions (except 15 01)');
INSERT INTO `ewc_level2` VALUES ('110', '20', '2', '20 02', 'garden and park wastes (including cemetery waste)');
INSERT INTO `ewc_level2` VALUES ('111', '20', '3', '20 03', 'other municipal wastes');

-- ----------------------------
-- Table structure for ewc_level3
-- ----------------------------
DROP TABLE IF EXISTS `ewc_level3`;
CREATE TABLE `ewc_level3` (
  `id` varchar(255) DEFAULT NULL,
  `parent_id` varchar(255) DEFAULT NULL,
  `EWC_id` varchar(255) DEFAULT NULL,
  `EWC_level3` varchar(255) DEFAULT NULL,
  `hazardous` varchar(255) DEFAULT NULL,
  `description` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of ewc_level3
-- ----------------------------
INSERT INTO `ewc_level3` VALUES ('591', '75', '17', '16 01 17', '1', 'ferrous metal');
INSERT INTO `ewc_level3` VALUES ('782', '107', '1', '19 12 01', '0', 'paper and cardboard');
INSERT INTO `ewc_level3` VALUES ('666', '89', '5', '17 04 05', '1', 'iron and steel');
INSERT INTO `ewc_level3` VALUES ('131', '20', '5', '06 01 05', '1', 'nitric acid and nitrous acid');
INSERT INTO `ewc_level3` VALUES ('833', '110', '2', '20 02 02', '0', 'soil and stones');
INSERT INTO `ewc_level3` VALUES ('683', '92', '2', '17 08 02', '1', 'gypsum-based construction materials other than those mentioned in 17 08 01');

-- ----------------------------
-- Table structure for ewc_level3_full
-- ----------------------------
DROP TABLE IF EXISTS `ewc_level3_full`;
CREATE TABLE `ewc_level3_full` (
  `id` varchar(255) DEFAULT NULL,
  `parent_id` varchar(255) DEFAULT NULL,
  `EWC_id` varchar(255) DEFAULT NULL,
  `EWC_level3` varchar(255) DEFAULT NULL,
  `hazardous` varchar(255) DEFAULT NULL,
  `description` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of ewc_level3_full
-- ----------------------------
INSERT INTO `ewc_level3_full` VALUES ('1', '1', '1', '01 01 01', '0', 'wastes from mineral metalliferous excavation');
INSERT INTO `ewc_level3_full` VALUES ('2', '1', '2', '01 01 02', '0', 'wastes from mineral non-metalliferous excavation');
INSERT INTO `ewc_level3_full` VALUES ('3', '2', '4', '01 03 04', '1', 'acid-generating tailings from processing of sulphide ore');
INSERT INTO `ewc_level3_full` VALUES ('4', '2', '5', '01 03 05', '1', 'other tailings containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('5', '2', '6', '01 03 06', '0', 'tailings other than those mentioned in 01 03 04 and 01 03 05');
INSERT INTO `ewc_level3_full` VALUES ('6', '2', '7', '01 03 07', '1', 'other wastes containing dangerous substances from physical and chemical processing of metalliferous minerals');
INSERT INTO `ewc_level3_full` VALUES ('7', '2', '8', '01 03 08', '0', 'dusty and powdery wastes other than those mentioned in 01 03 07');
INSERT INTO `ewc_level3_full` VALUES ('8', '2', '9', '01 03 09', '0', 'red mud from alumina production other than the wastes mentioned in 01 03 07');
INSERT INTO `ewc_level3_full` VALUES ('9', '2', '10', '01 03 10', '0', 'red mud from alumina production containing hazardous substances other than the wastes mentioned in 01 03 07');
INSERT INTO `ewc_level3_full` VALUES ('10', '2', '99', '01 03 99', '1', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('11', '3', '7', '01 04 07', '0', 'wastes containing dangerous substances from physical and chemical processing of non-metalliferous minerals');
INSERT INTO `ewc_level3_full` VALUES ('12', '3', '8', '01 04 08', '0', 'waste gravel and crushed rocks other than those mentioned in 01 04 07');
INSERT INTO `ewc_level3_full` VALUES ('13', '3', '9', '01 04 09', '0', 'waste sand and clays');
INSERT INTO `ewc_level3_full` VALUES ('14', '3', '10', '01 04 10', '0', 'dusty and powdery wastes other than those mentioned in 01 04 07');
INSERT INTO `ewc_level3_full` VALUES ('15', '3', '11', '01 04 11', '0', 'wastes from potash and rock salt processing other than those mentioned in 01 04 07');
INSERT INTO `ewc_level3_full` VALUES ('16', '3', '12', '01 04 12', '0', 'tailings and other wastes from washing and cleaning of minerals other than those mentioned in 01 04 07 and 01 04 11');
INSERT INTO `ewc_level3_full` VALUES ('17', '3', '13', '01 04 13', '0', 'wastes from stone cutting and sawing other than those mentioned in 01 04 07');
INSERT INTO `ewc_level3_full` VALUES ('18', '3', '99', '01 04 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('19', '4', '4', '01 05 04', '0', 'freshwater drilling muds and wastes');
INSERT INTO `ewc_level3_full` VALUES ('20', '4', '5', '01 05 05', '1', 'oil-containing drilling muds and wastes');
INSERT INTO `ewc_level3_full` VALUES ('21', '4', '6', '01 05 06', '1', 'drilling muds and other drilling wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('22', '4', '7', '01 05 07', '0', 'barite-containing drilling muds and wastes other than those mentioned in 01 05 05 and 01 05 06');
INSERT INTO `ewc_level3_full` VALUES ('23', '4', '8', '01 05 08', '0', 'chloride-containing drilling muds and wastes other than those mentioned in 01 05 05 and 01 05 06');
INSERT INTO `ewc_level3_full` VALUES ('24', '4', '99', '01 05 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('25', '5', '1', '02 01 01', '0', 'sludges from washing and cleaning');
INSERT INTO `ewc_level3_full` VALUES ('26', '5', '2', '02 01 02', '0', 'animal-tissue waste');
INSERT INTO `ewc_level3_full` VALUES ('27', '5', '3', '02 01 03', '0', 'plant-tissue waste');
INSERT INTO `ewc_level3_full` VALUES ('28', '5', '4', '02 01 04', '0', 'waste plastics (except packaging)');
INSERT INTO `ewc_level3_full` VALUES ('29', '5', '6', '02 01 06', '0', 'animal faeces, urine and manure (including spoiled straw), effluent, collected separately and treated off-site');
INSERT INTO `ewc_level3_full` VALUES ('30', '5', '7', '02 01 07', '0', 'wastes from forestry');
INSERT INTO `ewc_level3_full` VALUES ('31', '5', '8', '02 01 08', '1', 'agrochemical waste containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('32', '5', '9', '02 01 09', '0', 'agrochemical waste other than those mentioned in 02 01 08');
INSERT INTO `ewc_level3_full` VALUES ('33', '5', '10', '02 01 10', '0', 'waste metal');
INSERT INTO `ewc_level3_full` VALUES ('34', '5', '99', '02 01 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('35', '6', '1', '02 02 01', '0', 'sludges from washing and cleaning');
INSERT INTO `ewc_level3_full` VALUES ('36', '6', '2', '02 02 02', '0', 'animal-tissue waste');
INSERT INTO `ewc_level3_full` VALUES ('37', '6', '3', '02 02 03', '0', 'materials unsuitable for consumption or processing');
INSERT INTO `ewc_level3_full` VALUES ('38', '6', '4', '02 02 04', '0', 'sludges from on-site effluent treatment');
INSERT INTO `ewc_level3_full` VALUES ('39', '6', '99', '02 02 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('40', '7', '1', '02 03 01', '0', 'sludges from washing, cleaning, peeling, centrifuging and separation');
INSERT INTO `ewc_level3_full` VALUES ('41', '7', '2', '02 03 02', '0', 'wastes from preserving agents');
INSERT INTO `ewc_level3_full` VALUES ('42', '7', '3', '02 03 03', '0', 'wastes from solvent extraction');
INSERT INTO `ewc_level3_full` VALUES ('43', '7', '4', '02 03 04', '0', 'materials unsuitable for consumption or processing');
INSERT INTO `ewc_level3_full` VALUES ('44', '7', '5', '02 03 05', '0', 'sludges from on-site effluent treatment');
INSERT INTO `ewc_level3_full` VALUES ('45', '7', '99', '02 03 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('46', '8', '1', '02 04 01', '0', 'soil from cleaning and washing beet');
INSERT INTO `ewc_level3_full` VALUES ('47', '8', '2', '02 04 02', '0', 'off-specification calcium carbonate');
INSERT INTO `ewc_level3_full` VALUES ('48', '8', '3', '02 04 03', '0', 'sludges from on-site effluent treatment');
INSERT INTO `ewc_level3_full` VALUES ('49', '8', '99', '02 04 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('50', '9', '1', '02 05 01', '0', 'materials unsuitable for consumption or processing');
INSERT INTO `ewc_level3_full` VALUES ('51', '9', '2', '02 05 02', '0', 'sludges from on-site effluent treatment');
INSERT INTO `ewc_level3_full` VALUES ('52', '9', '99', '02 05 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('53', '10', '1', '02 06 01', '0', 'materials unsuitable for consumption or processing');
INSERT INTO `ewc_level3_full` VALUES ('54', '10', '2', '02 06 02', '0', 'wastes from preserving agents');
INSERT INTO `ewc_level3_full` VALUES ('55', '10', '3', '02 06 03', '0', 'sludges from on-site effluent treatment');
INSERT INTO `ewc_level3_full` VALUES ('56', '10', '99', '02 06 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('57', '11', '1', '02 07 01', '0', 'wastes from washing, cleaning and mechanical reduction of raw materials');
INSERT INTO `ewc_level3_full` VALUES ('58', '11', '2', '02 07 02', '0', 'wastes from spirits distillation');
INSERT INTO `ewc_level3_full` VALUES ('59', '11', '3', '02 07 03', '0', 'wastes from chemical treatment');
INSERT INTO `ewc_level3_full` VALUES ('60', '11', '4', '02 07 04', '0', 'materials unsuitable for consumption or processing');
INSERT INTO `ewc_level3_full` VALUES ('61', '11', '5', '02 07 05', '0', 'sludges from on-site effluent treatment');
INSERT INTO `ewc_level3_full` VALUES ('62', '11', '99', '02 07 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('63', '12', '1', '03 01 01', '0', 'waste bark and cork');
INSERT INTO `ewc_level3_full` VALUES ('64', '12', '4', '03 01 04', '1', 'sawdust, shavings, cuttings, wood, particle board and veneer containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('65', '12', '5', '03 01 05', '0', 'sawdust, shavings, cuttings, wood, particle board and veneer other than those mentioned in 03 01 04');
INSERT INTO `ewc_level3_full` VALUES ('66', '12', '99', '03 01 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('67', '13', '1', '03 02 01', '1', 'non-halogenated organic wood preservatives');
INSERT INTO `ewc_level3_full` VALUES ('68', '13', '2', '03 02 02', '1', 'organochlorinated wood preservatives');
INSERT INTO `ewc_level3_full` VALUES ('69', '13', '3', '03 02 03', '1', 'organometallic wood preservatives');
INSERT INTO `ewc_level3_full` VALUES ('70', '13', '4', '03 02 04', '1', 'inorganic wood preservatives');
INSERT INTO `ewc_level3_full` VALUES ('71', '13', '5', '03 02 05', '1', 'other wood preservatives containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('72', '13', '99', '03 02 99', '0', 'wood preservatives not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('73', '14', '1', '03 03 01', '0', 'waste bark and wood');
INSERT INTO `ewc_level3_full` VALUES ('74', '14', '2', '03 03 02', '0', 'green liquor sludge (from recovery of cooking liquor)');
INSERT INTO `ewc_level3_full` VALUES ('75', '14', '5', '03 03 05', '0', 'de-inking sludges from paper recycling');
INSERT INTO `ewc_level3_full` VALUES ('76', '14', '7', '03 03 07', '0', 'mechanically separated rejects from pulping of waste paper and cardboard');
INSERT INTO `ewc_level3_full` VALUES ('77', '14', '8', '03 03 08', '0', 'wastes from sorting of paper and cardboard destined for recycling');
INSERT INTO `ewc_level3_full` VALUES ('78', '14', '9', '03 03 09', '0', 'lime mud waste');
INSERT INTO `ewc_level3_full` VALUES ('79', '14', '10', '03 03 10', '0', 'fibre rejects, fibre-, filler- and coating-sludges from mechanical separation');
INSERT INTO `ewc_level3_full` VALUES ('80', '14', '11', '03 03 11', '0', 'sludges from on-site effluent treatment other than those mentioned in 03 03 10');
INSERT INTO `ewc_level3_full` VALUES ('81', '14', '99', '03 03 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('82', '15', '1', '04 01 01', '0', 'fleshings and lime split wastes');
INSERT INTO `ewc_level3_full` VALUES ('83', '15', '2', '04 01 02', '0', 'liming waste');
INSERT INTO `ewc_level3_full` VALUES ('84', '15', '3', '04 01 03', '1', 'degreasing wastes containing solvents without a liquid phase');
INSERT INTO `ewc_level3_full` VALUES ('85', '15', '4', '04 01 04', '0', 'tanning liquor containing chromium');
INSERT INTO `ewc_level3_full` VALUES ('86', '15', '5', '04 01 05', '0', 'tanning liquor free of chromium');
INSERT INTO `ewc_level3_full` VALUES ('87', '15', '6', '04 01 06', '0', 'sludges, in particular from on-site effluent treatment containing chromium');
INSERT INTO `ewc_level3_full` VALUES ('88', '15', '7', '04 01 07', '0', 'sludges, in particular from on-site effluent treatment free of chromium');
INSERT INTO `ewc_level3_full` VALUES ('89', '15', '8', '04 01 08', '0', 'waste tanned leather (blue sheetings, shavings, cuttings, buffing dust) containing chromium');
INSERT INTO `ewc_level3_full` VALUES ('90', '15', '9', '04 01 09', '0', 'wastes from dressing and finishing');
INSERT INTO `ewc_level3_full` VALUES ('91', '15', '99', '04 01 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('92', '16', '9', '04 02 09', '0', 'wastes from composite materials (impregnated textile, elastomer, plastomer)');
INSERT INTO `ewc_level3_full` VALUES ('93', '16', '10', '04 02 10', '0', 'organic matter from natural products (for example grease, wax)');
INSERT INTO `ewc_level3_full` VALUES ('94', '16', '14', '04 02 14', '1', 'wastes from finishing containing organic solvents');
INSERT INTO `ewc_level3_full` VALUES ('95', '16', '15', '04 02 15', '0', 'wastes from finishing other than those mentioned in 04 02 14');
INSERT INTO `ewc_level3_full` VALUES ('96', '16', '16', '04 02 16', '1', 'dyestuffs and pigments containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('97', '16', '17', '04 02 17', '0', 'dyestuffs and pigments other than those mentioned in 04 02 16');
INSERT INTO `ewc_level3_full` VALUES ('98', '16', '19', '04 02 19', '1', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('99', '16', '20', '04 02 20', '0', 'Sludges from on-site effluent treatment other than those mentioned in 04 02 19');
INSERT INTO `ewc_level3_full` VALUES ('100', '16', '21', '04 02 21', '0', 'wastes from unprocessed textile fibres');
INSERT INTO `ewc_level3_full` VALUES ('101', '16', '22', '04 02 22', '0', 'wastes from processed textile fibres');
INSERT INTO `ewc_level3_full` VALUES ('102', '16', '99', '04 02 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('103', '17', '2', '05 01 02', '1', 'desalter sludges');
INSERT INTO `ewc_level3_full` VALUES ('104', '17', '3', '05 01 03', '1', 'tank bottom sludges');
INSERT INTO `ewc_level3_full` VALUES ('105', '17', '4', '05 01 04', '1', 'acid alkyl sludges');
INSERT INTO `ewc_level3_full` VALUES ('106', '17', '5', '05 01 05', '1', 'oil spills');
INSERT INTO `ewc_level3_full` VALUES ('107', '17', '6', '05 01 06', '1', 'oily sludges from maintenance operations of the plant or equipment');
INSERT INTO `ewc_level3_full` VALUES ('108', '17', '7', '05 01 07', '1', 'acid tars');
INSERT INTO `ewc_level3_full` VALUES ('109', '17', '8', '05 01 08', '1', 'other tars');
INSERT INTO `ewc_level3_full` VALUES ('110', '17', '9', '05 01 09', '1', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('111', '17', '10', '05 01 10', '0', 'Sludges from on-site effluent treatment other than those mentioned in 05 01 09');
INSERT INTO `ewc_level3_full` VALUES ('112', '17', '11', '05 01 11', '1', 'wastes from cleaning of fuels with bases');
INSERT INTO `ewc_level3_full` VALUES ('113', '17', '12', '05 01 12', '1', 'oil containing acids');
INSERT INTO `ewc_level3_full` VALUES ('114', '17', '13', '05 01 13', '0', 'boiler feedwater sludges');
INSERT INTO `ewc_level3_full` VALUES ('115', '17', '14', '05 01 14', '0', 'wastes from cooling columns');
INSERT INTO `ewc_level3_full` VALUES ('116', '17', '15', '05 01 15', '1', 'spent filter clays');
INSERT INTO `ewc_level3_full` VALUES ('117', '17', '16', '05 01 16', '0', 'sulphur-containing wastes from petroleum desulphurisation');
INSERT INTO `ewc_level3_full` VALUES ('118', '17', '17', '05 01 17', '0', 'bitumen');
INSERT INTO `ewc_level3_full` VALUES ('119', '17', '99', '05 01 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('120', '18', '1', '05 06 01', '1', 'acid tars');
INSERT INTO `ewc_level3_full` VALUES ('121', '18', '3', '05 06 03', '1', 'other tars');
INSERT INTO `ewc_level3_full` VALUES ('122', '18', '4', '05 06 04', '0', 'waste from cooling columns');
INSERT INTO `ewc_level3_full` VALUES ('123', '18', '99', '05 06 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('124', '19', '1', '05 07 01', '1', 'wastes containing mercury');
INSERT INTO `ewc_level3_full` VALUES ('125', '19', '2', '05 07 02', '0', 'wastes containing sulphur');
INSERT INTO `ewc_level3_full` VALUES ('126', '19', '99', '05 07 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('127', '20', '1', '06 01 01', '1', 'sulphuric acid and sulphurous acid');
INSERT INTO `ewc_level3_full` VALUES ('128', '20', '2', '06 01 02', '1', 'hydrochloric acid');
INSERT INTO `ewc_level3_full` VALUES ('129', '20', '3', '06 01 03', '1', 'hydrofluoric acid');
INSERT INTO `ewc_level3_full` VALUES ('130', '20', '4', '06 01 04', '1', 'phosphoric and phosphorous acid');
INSERT INTO `ewc_level3_full` VALUES ('131', '20', '5', '06 01 05', '1', 'nitric acid and nitrous acid');
INSERT INTO `ewc_level3_full` VALUES ('132', '20', '6', '06 01 06', '1', 'other acids');
INSERT INTO `ewc_level3_full` VALUES ('133', '20', '99', '06 01 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('134', '21', '1', '06 02 01', '1', 'calcium hydroxide');
INSERT INTO `ewc_level3_full` VALUES ('135', '21', '3', '06 02 03', '1', 'ammonium hydroxide');
INSERT INTO `ewc_level3_full` VALUES ('136', '21', '4', '06 02 04', '1', 'sodium and potassium hydroxide');
INSERT INTO `ewc_level3_full` VALUES ('137', '21', '5', '06 02 05', '1', 'other bases');
INSERT INTO `ewc_level3_full` VALUES ('138', '21', '99', '06 02 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('139', '22', '11', '06 03 11', '1', 'solid salts and solutions containing cyanides');
INSERT INTO `ewc_level3_full` VALUES ('140', '22', '13', '06 03 13', '1', 'solid salts and solutions containing heavy metals');
INSERT INTO `ewc_level3_full` VALUES ('141', '22', '14', '06 03 14', '0', 'solid salts and solutions other than those mentioned in 06 03 11 and 06 03 13');
INSERT INTO `ewc_level3_full` VALUES ('142', '22', '15', '06 03 15', '1', 'metallic oxides containing heavy metals');
INSERT INTO `ewc_level3_full` VALUES ('143', '22', '16', '06 03 16', '0', 'metallic oxides other than those mentioned in 06 03 15');
INSERT INTO `ewc_level3_full` VALUES ('144', '22', '99', '06 03 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('145', '23', '3', '06 04 03', '1', 'wastes containing arsenic');
INSERT INTO `ewc_level3_full` VALUES ('146', '23', '4', '06 04 04', '1', 'wastes containing mercury');
INSERT INTO `ewc_level3_full` VALUES ('147', '23', '5', '06 04 05', '1', 'wastes containing other heavy metals');
INSERT INTO `ewc_level3_full` VALUES ('148', '23', '99', '06 04 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('149', '24', '2', '06 05 02', '1', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('150', '24', '3', '06 05 03', '0', 'sludges from on-site effluent treatment other than those mentioned in 06 05 02');
INSERT INTO `ewc_level3_full` VALUES ('151', '25', '2', '06 06 02', '1', 'wastes containing dangerous sulphides');
INSERT INTO `ewc_level3_full` VALUES ('152', '25', '3', '06 06 03', '0', 'wastes containing sulphides other than those mentioned in 06 06 02');
INSERT INTO `ewc_level3_full` VALUES ('153', '25', '99', '06 06 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('154', '26', '1', '06 07 01', '1', 'wastes containing asbestos from electrolysis');
INSERT INTO `ewc_level3_full` VALUES ('155', '26', '2', '06 07 02', '1', 'activated carbon from chlorine production');
INSERT INTO `ewc_level3_full` VALUES ('156', '26', '3', '06 07 03', '1', 'barium sulphate sludge containing mercury');
INSERT INTO `ewc_level3_full` VALUES ('157', '26', '4', '06 07 04', '1', 'solutions and acids, for example contact acid');
INSERT INTO `ewc_level3_full` VALUES ('158', '26', '99', '06 07 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('159', '27', '2', '06 08 02', '1', 'wastes containing chlorosilanes');
INSERT INTO `ewc_level3_full` VALUES ('160', '27', '99', '06 08 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('161', '28', '2', '06 09 02', '0', 'phosphorous slag');
INSERT INTO `ewc_level3_full` VALUES ('162', '28', '3', '06 09 03', '1', 'calcium-based reaction wastes containing or contaminated with dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('163', '28', '4', '06 09 04', '0', 'calcium-based reaction wastes other than those mentioned in 06 09 03');
INSERT INTO `ewc_level3_full` VALUES ('164', '28', '99', '06 09 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('165', '29', '2', '06 10 02', '1', 'wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('166', '29', '99', '06 10 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('167', '30', '1', '06 11 01', '0', 'calcium-based reaction wastes from titanium dioxide production');
INSERT INTO `ewc_level3_full` VALUES ('168', '30', '99', '06 11 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('169', '31', '1', '06 13 01', '1', 'inorganic plant protection products, wood-preserving agents and other biocides.');
INSERT INTO `ewc_level3_full` VALUES ('170', '31', '2', '06 13 02', '1', 'spent activated carbon (except 06 07 02)');
INSERT INTO `ewc_level3_full` VALUES ('171', '31', '3', '06 13 03', '0', 'carbon black');
INSERT INTO `ewc_level3_full` VALUES ('172', '31', '4', '06 13 04', '1', 'wastes from asbestos processing');
INSERT INTO `ewc_level3_full` VALUES ('173', '31', '5', '06 13 05', '1', 'soot');
INSERT INTO `ewc_level3_full` VALUES ('174', '31', '99', '06 13 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('175', '32', '1', '07 01 01', '1', 'aqueous washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('176', '32', '3', '07 01 03', '1', 'organic halogenated solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('177', '32', '4', '07 01 04', '1', 'other organic solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('178', '32', '7', '07 01 07', '1', 'halogenated still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('179', '32', '8', '07 01 08', '1', 'other still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('180', '32', '9', '07 01 09', '1', 'halogenated filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('181', '32', '10', '07 01 10', '1', 'other filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('182', '32', '11', '07 01 11', '1', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('183', '32', '12', '07 01 12', '0', 'sludges from on-site effluent treatment other than those mentioned in 07 01 11');
INSERT INTO `ewc_level3_full` VALUES ('184', '32', '99', '07 01 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('185', '33', '1', '07 02 01', '1', 'aqueous washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('186', '33', '3', '07 02 03', '1', 'organic halogenated solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('187', '33', '4', '07 02 04', '1', 'other organic solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('188', '33', '7', '07 02 07', '1', 'halogenated still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('189', '33', '8', '07 02 08', '1', 'other still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('190', '33', '9', '07 02 09', '1', 'halogenated filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('191', '33', '10', '07 02 10', '1', 'other filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('192', '33', '11', '07 02 11', '1', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('193', '33', '12', '07 02 12', '0', 'sludges from on-site effluent treatment other than those mentioned in 07 02 11');
INSERT INTO `ewc_level3_full` VALUES ('194', '33', '13', '07 02 13', '0', 'waste plastic');
INSERT INTO `ewc_level3_full` VALUES ('195', '33', '14', '07 02 14', '1', 'wastes from additives containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('196', '33', '15', '07 02 15', '0', 'wastes from additives other than those mentioned in 07 02 14');
INSERT INTO `ewc_level3_full` VALUES ('197', '33', '16', '07 02 16', '1', 'wastes containing silicones');
INSERT INTO `ewc_level3_full` VALUES ('198', '33', '17', '07 02 17', '0', 'wastes containing silicones other than those mentioned in 07 02 16');
INSERT INTO `ewc_level3_full` VALUES ('199', '33', '99', '07 02 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('200', '34', '1', '07 03 01', '1', 'aqueous washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('201', '34', '3', '07 03 03', '1', 'organic halogenated solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('202', '34', '4', '07 03 04', '1', 'other organic solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('203', '34', '7', '07 03 07', '1', 'halogenated still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('204', '34', '8', '07 03 08', '1', 'other still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('205', '34', '9', '07 03 09', '1', 'halogenated filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('206', '34', '10', '07 03 10', '1', 'other filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('207', '34', '11', '07 03 11', '1', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('208', '34', '12', '07 03 12', '0', 'sludges from on-site effluent treatment other than those mentioned in 07 03 11');
INSERT INTO `ewc_level3_full` VALUES ('209', '34', '99', '07 03 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('210', '35', '1', '07 04 01', '1', 'aqueous washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('211', '35', '3', '07 04 03', '1', 'organic halogenated solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('212', '35', '4', '07 04 04', '1', 'other organic solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('213', '35', '7', '07 04 07', '1', 'halogenated still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('214', '35', '8', '07 04 08', '1', 'other still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('215', '35', '9', '07 04 09', '1', 'halogenated filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('216', '35', '10', '07 04 10', '1', 'other filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('217', '35', '11', '07 04 11', '1', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('218', '35', '12', '07 04 12', '0', 'sludges from on-site effluent treatment other than those mentioned in 07 04 11');
INSERT INTO `ewc_level3_full` VALUES ('219', '35', '13', '07 04 13', '1', 'solid wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('220', '35', '99', '07 04 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('221', '36', '1', '07 05 01', '1', 'aqueous washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('222', '36', '3', '07 05 03', '1', 'organic halogenated solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('223', '36', '4', '07 05 04', '1', 'other organic solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('224', '36', '7', '07 05 07', '1', 'halogenated still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('225', '36', '8', '07 05 08', '1', 'other still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('226', '36', '9', '07 05 09', '1', 'halogenated filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('227', '36', '10', '07 05 10', '1', 'other filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('228', '36', '11', '07 05 11', '1', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('229', '36', '12', '07 05 12', '0', 'sludges from on-site effluent treatment other than those mentioned in 07 05 11');
INSERT INTO `ewc_level3_full` VALUES ('230', '36', '13', '07 05 13', '1', 'solid wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('231', '36', '14', '07 05 14', '0', 'solid wastes other than those mentioned in 07 05 13');
INSERT INTO `ewc_level3_full` VALUES ('232', '36', '99', '07 05 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('233', '37', '1', '07 06 01', '1', 'aqueous washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('234', '37', '3', '07 06 03', '1', 'organic halogenated solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('235', '37', '4', '07 06 04', '1', 'other organic solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('236', '37', '7', '07 06 07', '1', 'halogenated still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('237', '37', '8', '07 06 08', '1', 'other still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('238', '37', '9', '07 06 09', '1', 'halogenated filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('239', '37', '10', '07 06 10', '1', 'other filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('240', '37', '11', '07 06 11', '1', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('241', '37', '12', '07 06 12', '0', 'sludges from on-site effluent treatment other than those mentioned in 07 06 11');
INSERT INTO `ewc_level3_full` VALUES ('242', '37', '99', '07 06 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('243', '38', '1', '07 07 01', '1', 'aqueous washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('244', '38', '3', '07 07 03', '1', 'organic halogenated solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('245', '38', '4', '07 07 04', '1', 'other organic solvents, washing liquids and mother liquors');
INSERT INTO `ewc_level3_full` VALUES ('246', '38', '7', '07 07 07', '1', 'halogenated still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('247', '38', '8', '07 07 08', '1', 'other still bottoms and reaction residues');
INSERT INTO `ewc_level3_full` VALUES ('248', '38', '9', '07 07 09', '1', 'halogenated filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('249', '38', '10', '07 07 10', '1', 'other filter cakes and spent absorbents');
INSERT INTO `ewc_level3_full` VALUES ('250', '38', '11', '07 07 11', '1', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('251', '38', '12', '07 07 12', '0', 'sludges from on-site effluent treatment other than those mentioned in 07 07 11');
INSERT INTO `ewc_level3_full` VALUES ('252', '38', '99', '07 07 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('253', '39', '11', '08 01 11', '1', 'waste paint and varnish containing organic solvents or other dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('254', '39', '12', '08 01 12', '0', 'waste paint and varnish other than those mentioned in 08 01 11');
INSERT INTO `ewc_level3_full` VALUES ('255', '39', '13', '08 01 13', '1', 'sludges from paint or varnish containing organic solvents or other dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('256', '39', '14', '08 01 14', '0', 'sludges from paint or varnish other than those mentioned in 08 01 13');
INSERT INTO `ewc_level3_full` VALUES ('257', '39', '15', '08 01 15', '1', 'aqueous sludges containing paint or varnish containing organic solvents or other dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('258', '39', '16', '08 01 16', '0', 'aqueous sludges containing paint or varnish other than those mentioned in 08 01 15');
INSERT INTO `ewc_level3_full` VALUES ('259', '39', '17', '08 01 17', '1', 'wastes from paint or varnish removal containing organic solvents or other dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('260', '39', '18', '08 01 18', '0', 'wastes from paint or varnish removal other than those mentioned in 08 01 17');
INSERT INTO `ewc_level3_full` VALUES ('261', '39', '19', '08 01 19', '1', 'aqueous suspensions containing paint or varnish containing organic solvents or other dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('262', '39', '20', '08 01 20', '0', 'aqueous suspensions containing paint or varnish other than those mentioned in 08 01 19');
INSERT INTO `ewc_level3_full` VALUES ('263', '39', '21', '08 01 21', '1', 'waste paint or varnish remover');
INSERT INTO `ewc_level3_full` VALUES ('264', '39', '99', '08 01 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('265', '40', '1', '08 02 01', '0', 'waste coating powders');
INSERT INTO `ewc_level3_full` VALUES ('266', '40', '2', '08 02 02', '0', 'aqueous sludges containing ceramic materials');
INSERT INTO `ewc_level3_full` VALUES ('267', '40', '3', '08 02 03', '0', 'aqueous suspensions containing ceramic materials');
INSERT INTO `ewc_level3_full` VALUES ('268', '40', '99', '08 02 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('269', '41', '7', '08 03 07', '0', 'aqueous sludges containing ink');
INSERT INTO `ewc_level3_full` VALUES ('270', '41', '8', '08 03 08', '0', 'aqueous liquid waste containing ink');
INSERT INTO `ewc_level3_full` VALUES ('271', '41', '12', '08 03 12', '1', 'waste ink containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('272', '41', '13', '08 03 13', '0', 'waste ink other than those mentioned in 08 03 12');
INSERT INTO `ewc_level3_full` VALUES ('273', '41', '14', '08 03 14', '1', 'ink sludges containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('274', '41', '15', '08 03 15', '0', 'ink sludges other than those mentioned in 08 03 14');
INSERT INTO `ewc_level3_full` VALUES ('275', '41', '16', '08 03 16', '1', 'waste etching solutions');
INSERT INTO `ewc_level3_full` VALUES ('276', '41', '17', '08 03 17', '1', 'waste printing toner containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('277', '41', '18', '08 03 18', '0', 'waste printing toner other than those mentioned in 08 03 17');
INSERT INTO `ewc_level3_full` VALUES ('278', '41', '19', '08 03 19', '1', 'disperse oil');
INSERT INTO `ewc_level3_full` VALUES ('279', '41', '99', '08 03 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('280', '42', '9', '08 04 09', '1', 'waste adhesives and sealants containing organic solvents or other dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('281', '42', '10', '08 04 10', '0', 'waste adhesives and sealants other than those mentioned in 08 04 09');
INSERT INTO `ewc_level3_full` VALUES ('282', '42', '11', '08 04 11', '1', 'adhesive and sealant sludges containing organic solvents or other dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('283', '42', '12', '08 04 12', '0', 'adhesive and sealant sludges other than those mentioned in 08 04 11');
INSERT INTO `ewc_level3_full` VALUES ('284', '42', '13', '08 04 13', '1', 'aqueous sludges containing adhesives or sealants containing organic solvents or other dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('285', '42', '14', '08 04 14', '0', 'aqueous sludges containing adhesives or sealants other than those mentioned in 08 04 13');
INSERT INTO `ewc_level3_full` VALUES ('286', '42', '15', '08 04 15', '1', 'aqueous liquid waste containing adhesives or sealants containing organic solvents or other dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('287', '42', '16', '08 04 16', '0', 'aqueous liquid waste containing adhesives or sealants other than those mentioned in 08 04 15');
INSERT INTO `ewc_level3_full` VALUES ('288', '42', '17', '08 04 17', '1', 'rosin oil');
INSERT INTO `ewc_level3_full` VALUES ('289', '42', '99', '08 04 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('290', '43', '1', '08 05 01', '1', 'waste isocyanates');
INSERT INTO `ewc_level3_full` VALUES ('291', '44', '1', '09 01 01', '1', 'water-based developer and activator solutions');
INSERT INTO `ewc_level3_full` VALUES ('292', '44', '2', '09 01 02', '1', 'water-based offset plate developer solutions');
INSERT INTO `ewc_level3_full` VALUES ('293', '44', '3', '09 01 03', '1', 'solvent-based developer solutions');
INSERT INTO `ewc_level3_full` VALUES ('294', '44', '4', '09 01 04', '1', 'fixer solutions');
INSERT INTO `ewc_level3_full` VALUES ('295', '44', '5', '09 01 05', '1', 'bleach solutions and bleach fixer solutions');
INSERT INTO `ewc_level3_full` VALUES ('296', '44', '6', '09 01 06', '1', 'wastes containing silver from on-site treatment of photographic wastes');
INSERT INTO `ewc_level3_full` VALUES ('297', '44', '7', '09 01 07', '0', 'photographic film and paper containing silver or silver compounds');
INSERT INTO `ewc_level3_full` VALUES ('298', '44', '8', '09 01 08', '0', 'photographic film and paper free of silver or silver compounds');
INSERT INTO `ewc_level3_full` VALUES ('299', '44', '10', '09 01 10', '0', 'single-use cameras without batteries');
INSERT INTO `ewc_level3_full` VALUES ('300', '44', '11', '09 01 11', '1', 'single-use cameras containing batteries included in 16 06 01, 16 06 02 or 16 06 03');
INSERT INTO `ewc_level3_full` VALUES ('301', '44', '12', '09 01 12', '0', 'single-use cameras containing batteries other than those mentioned in 09 01 11');
INSERT INTO `ewc_level3_full` VALUES ('302', '44', '13', '09 01 13', '1', 'aqueous liquid waste from on-site reclamation of silver other than those mentioned in 09 01 06');
INSERT INTO `ewc_level3_full` VALUES ('303', '44', '99', '09 01 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('304', '45', '1', '10 01 01', '0', 'bottom ash, slag and boiler dust (excluding boiler dust mentioned in 10 01 04)');
INSERT INTO `ewc_level3_full` VALUES ('305', '45', '2', '10 01 02', '0', 'coal fly ash');
INSERT INTO `ewc_level3_full` VALUES ('306', '45', '3', '10 01 03', '0', 'fly ash from peat and untreated wood');
INSERT INTO `ewc_level3_full` VALUES ('307', '45', '4', '10 01 04', '1', 'oil fly ash and boiler dust');
INSERT INTO `ewc_level3_full` VALUES ('308', '45', '5', '10 01 05', '0', 'Calcium-based reaction wastes from flue-gas desulphurisation in solid form');
INSERT INTO `ewc_level3_full` VALUES ('309', '45', '7', '10 01 07', '0', 'calcium-based reaction wastes from flue-gas desulphurisation in sludge form');
INSERT INTO `ewc_level3_full` VALUES ('310', '45', '9', '10 01 09', '1', 'sulphuric acid');
INSERT INTO `ewc_level3_full` VALUES ('311', '45', '13', '10 01 13', '1', 'fly ash from emulsified hydrocarbons used as fuel');
INSERT INTO `ewc_level3_full` VALUES ('312', '45', '14', '10 01 14', '1', 'bottom ash, slag and boiler dust from co-incineration containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('313', '45', '15', '10 01 15', '0', 'Bottom ash, slag and boiler dust from co-incineration other than those mentioned in 10 01 14');
INSERT INTO `ewc_level3_full` VALUES ('314', '45', '16', '10 01 16', '1', 'fly ash from co-incineration containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('315', '45', '17', '10 01 17', '0', 'fly ash from co-incineration other than those mentioned in 10 01 16');
INSERT INTO `ewc_level3_full` VALUES ('316', '45', '18', '10 01 18', '1', 'wastes from gas cleaning containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('317', '45', '19', '10 01 19', '0', 'wastes from gas cleaning other than those mentioned in 10 01 05, 10 01 07 and 10 01 18');
INSERT INTO `ewc_level3_full` VALUES ('318', '45', '20', '10 01 20', '1', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('319', '45', '21', '10 01 21', '0', 'sludges from on-site effluent treatment other than those mentioned in 10 01 20');
INSERT INTO `ewc_level3_full` VALUES ('320', '45', '22', '10 01 22', '1', 'aqueous sludges from boiler cleansing containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('321', '45', '23', '10 01 23', '0', 'aqueous sludges from boiler cleansing other than those mentioned in 10 01 22');
INSERT INTO `ewc_level3_full` VALUES ('322', '45', '24', '10 01 24', '0', 'sands from fluidised beds');
INSERT INTO `ewc_level3_full` VALUES ('323', '45', '25', '10 01 25', '0', 'wastes from fuel storage and preparation of coal-fired power plants');
INSERT INTO `ewc_level3_full` VALUES ('324', '45', '26', '10 01 26', '0', 'wastes from cooling-water treatment');
INSERT INTO `ewc_level3_full` VALUES ('325', '45', '99', '10 01 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('326', '46', '1', '10 02 01', '0', 'wastes from the processing of slag');
INSERT INTO `ewc_level3_full` VALUES ('327', '46', '2', '10 02 02', '0', 'unprocessed slag');
INSERT INTO `ewc_level3_full` VALUES ('328', '46', '7', '10 02 07', '1', 'solid wastes from gas treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('329', '46', '8', '10 02 08', '0', 'solid wastes from gas treatment other than those mentioned in 10 02 07');
INSERT INTO `ewc_level3_full` VALUES ('330', '46', '10', '10 02 10', '0', 'mill scales');
INSERT INTO `ewc_level3_full` VALUES ('331', '46', '11', '10 02 11', '1', 'wastes from cooling-water treatment containing oil');
INSERT INTO `ewc_level3_full` VALUES ('332', '46', '12', '10 02 12', '0', 'wastes from cooling-water treatment other than those mentioned in 10 02 11');
INSERT INTO `ewc_level3_full` VALUES ('333', '46', '13', '10 02 13', '1', 'sludges and filter cakes from gas treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('334', '46', '14', '10 02 14', '0', 'sludges and filter cakes from gas treatment other than those mentioned in 10 02 13');
INSERT INTO `ewc_level3_full` VALUES ('335', '46', '15', '10 02 15', '0', 'other sludges and filter cakes');
INSERT INTO `ewc_level3_full` VALUES ('336', '46', '99', '10 02 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('337', '47', '2', '10 03 02', '0', 'anode scraps');
INSERT INTO `ewc_level3_full` VALUES ('338', '47', '4', '10 03 04', '1', 'primary production slags');
INSERT INTO `ewc_level3_full` VALUES ('339', '47', '5', '10 03 05', '0', 'waste alumina');
INSERT INTO `ewc_level3_full` VALUES ('340', '47', '8', '10 03 08', '1', 'salt slags from secondary production');
INSERT INTO `ewc_level3_full` VALUES ('341', '47', '9', '10 03 09', '1', 'black drosses from secondary production');
INSERT INTO `ewc_level3_full` VALUES ('342', '47', '15', '10 03 15', '1', 'skimmings that are flammable or emit, upon contact with water, flammable gases in dangerous quantities');
INSERT INTO `ewc_level3_full` VALUES ('343', '47', '16', '10 03 16', '0', 'skimmings other than those mentioned in 10 03 15');
INSERT INTO `ewc_level3_full` VALUES ('344', '47', '17', '10 03 17', '1', 'tar-containing wastes from anode manufacture');
INSERT INTO `ewc_level3_full` VALUES ('345', '47', '18', '10 03 18', '0', 'carbon-containing wastes from anode manufacture other than those mentioned in 10 03 17');
INSERT INTO `ewc_level3_full` VALUES ('346', '47', '19', '10 03 19', '1', 'flue-gas dust containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('347', '47', '20', '10 03 20', '0', 'flue-gas dust other than those mentioned in 10 03 19');
INSERT INTO `ewc_level3_full` VALUES ('348', '47', '21', '10 03 21', '1', 'other particulates and dust (including ball-mill dust) containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('349', '47', '22', '10 03 22', '0', 'other particulates and dust (including ball-mill dust) other than those mentioned in 10 03 21');
INSERT INTO `ewc_level3_full` VALUES ('350', '47', '23', '10 03 23', '1', 'solid wastes from gas treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('351', '47', '24', '10 03 24', '0', 'solid wastes from gas treatment other than those mentioned in 10 03 23');
INSERT INTO `ewc_level3_full` VALUES ('352', '47', '25', '10 03 25', '1', 'sludges and filter cakes from gas treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('353', '47', '26', '10 03 26', '0', 'sludges and filter cakes from gas treatment other than those mentioned in 10 03 25');
INSERT INTO `ewc_level3_full` VALUES ('354', '47', '27', '10 03 27', '1', 'wastes from cooling-water treatment containing oil');
INSERT INTO `ewc_level3_full` VALUES ('355', '47', '28', '10 03 28', '0', 'wastes from cooling-water treatment other than those mentioned in 10 03 27');
INSERT INTO `ewc_level3_full` VALUES ('356', '47', '29', '10 03 29', '1', 'wastes from treatment of salt slags and black drosses containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('357', '47', '30', '10 03 30', '0', 'wastes from treatment of salt slags and black drosses other than those mentioned in 10 03 29');
INSERT INTO `ewc_level3_full` VALUES ('358', '47', '99', '10 03 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('359', '48', '1', '10 04 01', '1', 'slags from primary and secondary production');
INSERT INTO `ewc_level3_full` VALUES ('360', '48', '2', '10 04 02', '1', 'dross and skimmings from primary and secondary production');
INSERT INTO `ewc_level3_full` VALUES ('361', '48', '3', '10 04 03', '1', 'calcium arsenate');
INSERT INTO `ewc_level3_full` VALUES ('362', '48', '4', '10 04 04', '1', 'flue-gas dust');
INSERT INTO `ewc_level3_full` VALUES ('363', '48', '5', '10 04 05', '1', 'other particulates and dust');
INSERT INTO `ewc_level3_full` VALUES ('364', '48', '6', '10 04 06', '1', 'solid wastes from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('365', '48', '7', '10 04 07', '1', 'sludges and filter cakes from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('366', '48', '9', '10 04 09', '1', 'wastes from cooling-water treatment containing oil');
INSERT INTO `ewc_level3_full` VALUES ('367', '48', '10', '10 04 10', '0', 'wastes from cooling-water treatment other than those mentioned in 10 04 09');
INSERT INTO `ewc_level3_full` VALUES ('368', '48', '99', '10 04 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('369', '49', '1', '10 05 01', '0', 'slags from primary and secondary production');
INSERT INTO `ewc_level3_full` VALUES ('370', '49', '3', '10 05 03', '1', 'flue-gas dust');
INSERT INTO `ewc_level3_full` VALUES ('371', '49', '4', '10 05 04', '0', 'other particulates and dust');
INSERT INTO `ewc_level3_full` VALUES ('372', '49', '5', '10 05 05', '1', 'solid waste from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('373', '49', '6', '10 05 06', '1', 'sludges and filter cakes from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('374', '49', '8', '10 05 08', '1', 'wastes from cooling-water treatment containing oil');
INSERT INTO `ewc_level3_full` VALUES ('375', '49', '9', '10 05 09', '0', 'wastes from cooling-water treatment other than those mentioned in 10 05 08');
INSERT INTO `ewc_level3_full` VALUES ('376', '49', '10', '10 05 10', '1', 'dross and skimmings that are flammable or emit, upon contact with water, flammable gases in dangerous quantities');
INSERT INTO `ewc_level3_full` VALUES ('377', '49', '11', '10 05 11', '0', 'dross and skimmings other than those mentioned in 10 05 10');
INSERT INTO `ewc_level3_full` VALUES ('378', '49', '99', '10 05 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('379', '50', '1', '10 06 01', '0', 'slags from primary and secondary production');
INSERT INTO `ewc_level3_full` VALUES ('380', '50', '2', '10 06 02', '0', 'dross and skimmings from primary and secondary production');
INSERT INTO `ewc_level3_full` VALUES ('381', '50', '3', '10 06 03', '1', 'flue-gas dust');
INSERT INTO `ewc_level3_full` VALUES ('382', '50', '4', '10 06 04', '0', 'other particulates and dust');
INSERT INTO `ewc_level3_full` VALUES ('383', '50', '6', '10 06 06', '1', 'solid wastes from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('384', '50', '7', '10 06 07', '1', 'sludges and filter cakes from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('385', '50', '9', '10 06 09', '1', 'wastes from cooling-water treatment containing oil');
INSERT INTO `ewc_level3_full` VALUES ('386', '50', '10', '10 06 10', '0', 'wastes from cooling-water treatment other than those mentioned in 10 06 09');
INSERT INTO `ewc_level3_full` VALUES ('387', '50', '99', '10 06 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('388', '51', '1', '10 07 01', '0', 'slags from primary and secondary production');
INSERT INTO `ewc_level3_full` VALUES ('389', '51', '2', '10 07 02', '0', 'dross and skimmings from primary and secondary production');
INSERT INTO `ewc_level3_full` VALUES ('390', '51', '3', '10 07 03', '0', 'solid wastes from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('391', '51', '4', '10 07 04', '0', 'other particulates and dust');
INSERT INTO `ewc_level3_full` VALUES ('392', '51', '5', '10 07 05', '0', 'sludges and filter cakes from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('393', '51', '7', '10 07 07', '1', 'wastes from cooling-water treatment containing oil');
INSERT INTO `ewc_level3_full` VALUES ('394', '51', '8', '10 07 08', '0', 'wastes from cooling-water treatment other than those mentioned in 10 07 07');
INSERT INTO `ewc_level3_full` VALUES ('395', '51', '99', '10 07 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('396', '52', '4', '10 08 04', '0', 'particulates and dust');
INSERT INTO `ewc_level3_full` VALUES ('397', '52', '8', '10 08 08', '1', 'salt slag from primary and secondary production');
INSERT INTO `ewc_level3_full` VALUES ('398', '52', '9', '10 08 09', '0', 'other slags');
INSERT INTO `ewc_level3_full` VALUES ('399', '52', '10', '10 08 10', '1', 'dross and skimmings that are flammable or emit, upon contact with water, flammable gases in dangerous quantities');
INSERT INTO `ewc_level3_full` VALUES ('400', '52', '11', '10 08 11', '0', 'dross and skimmings other than those mentioned in 10 08 10');
INSERT INTO `ewc_level3_full` VALUES ('401', '52', '12', '10 08 12', '1', 'tar-containing wastes from anode manufacture');
INSERT INTO `ewc_level3_full` VALUES ('402', '52', '13', '10 08 13', '0', 'carbon-containing wastes from anode manufacture other than those mentioned in 10 08 12');
INSERT INTO `ewc_level3_full` VALUES ('403', '52', '14', '10 08 14', '0', 'anode scrap');
INSERT INTO `ewc_level3_full` VALUES ('404', '52', '15', '10 08 15', '1', 'flue-gas dust containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('405', '52', '16', '10 08 16', '0', 'flue-gas dust other than those mentioned in 10 08 15');
INSERT INTO `ewc_level3_full` VALUES ('406', '52', '17', '10 08 17', '1', 'sludges and filter cakes from flue-gas treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('407', '52', '18', '10 08 18', '0', 'sludges and filter cakes from flue-gas treatment other than those mentioned in 10 08 17');
INSERT INTO `ewc_level3_full` VALUES ('408', '52', '19', '10 08 19', '1', 'wastes from cooling-water treatment containing oil');
INSERT INTO `ewc_level3_full` VALUES ('409', '52', '20', '10 08 20', '0', 'wastes from cooling-water treatment other than those mentioned in 10 08 19');
INSERT INTO `ewc_level3_full` VALUES ('410', '52', '99', '10 08 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('411', '53', '3', '10 09 03', '0', 'furnace slag');
INSERT INTO `ewc_level3_full` VALUES ('412', '53', '5', '10 09 05', '0', 'casting cores and moulds which have not undergone pouring containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('413', '53', '6', '10 09 06', '1', 'casting cores and moulds which have not undergone pouring other than those mentioned in 10 09 05');
INSERT INTO `ewc_level3_full` VALUES ('414', '53', '7', '10 09 07', '0', 'casting cores and moulds which have undergone pouring containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('415', '53', '8', '10 09 08', '1', 'casting cores and moulds which have undergone pouring other than those mentioned in 10 09 07');
INSERT INTO `ewc_level3_full` VALUES ('416', '53', '9', '10 09 09', '0', 'flue-gas dust containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('417', '53', '10', '10 09 10', '1', 'flue-gas dust other than those mentioned in 10 09 09');
INSERT INTO `ewc_level3_full` VALUES ('418', '53', '11', '10 09 11', '0', 'other particulates containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('419', '53', '12', '10 09 12', '1', 'other particulates other than those mentioned in 10 09 11');
INSERT INTO `ewc_level3_full` VALUES ('420', '53', '13', '10 09 13', '0', 'waste binders containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('421', '53', '14', '10 09 14', '1', 'waste binders other than those mentioned in 10 09 13');
INSERT INTO `ewc_level3_full` VALUES ('422', '53', '15', '10 09 15', '0', 'waste crack-indicating agent containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('423', '53', '16', '10 09 16', '1', 'waste crack-indicating agent other than those mentioned in 10 09 15');
INSERT INTO `ewc_level3_full` VALUES ('424', '53', '99', '10 09 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('425', '54', '3', '10 10 03', '0', 'furnace slag');
INSERT INTO `ewc_level3_full` VALUES ('426', '54', '5', '10 10 05', '0', 'casting cores and moulds which have not undergone pouring, containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('427', '54', '6', '10 10 06', '1', 'casting cores and moulds which have not undergone pouring, other than those mentioned in 10 10 05');
INSERT INTO `ewc_level3_full` VALUES ('428', '54', '7', '10 10 07', '0', 'casting cores and moulds which have undergone pouring, containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('429', '54', '8', '10 10 08', '1', 'casting cores and moulds which have undergone pouring, other than those mentioned in 10 10 07');
INSERT INTO `ewc_level3_full` VALUES ('430', '54', '9', '10 10 09', '0', 'flue-gas dust containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('431', '54', '10', '10 10 10', '1', 'flue-gas dust other than those mentioned in 10 10 09');
INSERT INTO `ewc_level3_full` VALUES ('432', '54', '11', '10 10 11', '0', 'other particulates containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('433', '54', '12', '10 10 12', '1', 'other particulates other than those mentioned in 10 10 11');
INSERT INTO `ewc_level3_full` VALUES ('434', '54', '13', '10 10 13', '0', 'waste binders containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('435', '54', '14', '10 10 14', '1', 'waste binders other than those mentioned in 10 10 13');
INSERT INTO `ewc_level3_full` VALUES ('436', '54', '15', '10 10 15', '0', 'waste crack-indicating agent containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('437', '54', '16', '10 10 16', '1', 'waste crack-indicating agent other than those mentioned in 10 10 15');
INSERT INTO `ewc_level3_full` VALUES ('438', '54', '99', '10 10 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('439', '55', '3', '10 11 03', '0', 'waste glass-based fibrous materials');
INSERT INTO `ewc_level3_full` VALUES ('440', '55', '5', '10 11 05', '0', 'particulates and dust');
INSERT INTO `ewc_level3_full` VALUES ('441', '55', '9', '10 11 09', '0', 'waste preparation mixture before thermal processing, containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('442', '55', '10', '10 11 10', '1', 'waste preparation mixture before thermal processing, other than those mentioned in 10 11 09');
INSERT INTO `ewc_level3_full` VALUES ('443', '55', '11', '10 11 11', '0', 'waste glass in small particles and glass powder containing heavy metals (for example from cathode ray tubes)');
INSERT INTO `ewc_level3_full` VALUES ('444', '55', '12', '10 11 12', '1', 'waste glass other than those mentioned in 10 11 11');
INSERT INTO `ewc_level3_full` VALUES ('445', '55', '13', '10 11 13', '0', 'glass-polishing and -grinding sludge containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('446', '55', '14', '10 11 14', '1', 'glass-polishing and -grinding sludge other than those mentioned in 10 11 13');
INSERT INTO `ewc_level3_full` VALUES ('447', '55', '15', '10 11 15', '0', 'solid wastes from flue-gas treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('448', '55', '16', '10 11 16', '1', 'solid wastes from flue-gas treatment other than those mentioned in 10 11 15');
INSERT INTO `ewc_level3_full` VALUES ('449', '55', '17', '10 11 17', '0', 'sludges and filter cakes from flue-gas treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('450', '55', '18', '10 11 18', '1', 'sludges and filter cakes from flue-gas treatment other than those mentioned in 10 11 17');
INSERT INTO `ewc_level3_full` VALUES ('451', '55', '19', '10 11 19', '0', 'solid wastes from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('452', '55', '20', '10 11 20', '1', 'solid wastes from on-site effluent treatment other than those mentioned in 10 11 19');
INSERT INTO `ewc_level3_full` VALUES ('453', '55', '99', '10 11 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('454', '56', '1', '10 12 01', '0', 'waste preparation mixture before thermal processing');
INSERT INTO `ewc_level3_full` VALUES ('455', '56', '3', '10 12 03', '0', 'particulates and dust');
INSERT INTO `ewc_level3_full` VALUES ('456', '56', '5', '10 12 05', '0', 'sludges and filter cakes from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('457', '56', '6', '10 12 06', '0', 'discarded moulds');
INSERT INTO `ewc_level3_full` VALUES ('458', '56', '8', '10 12 08', '0', 'waste ceramics, bricks, tiles and construction products (after thermal processing)');
INSERT INTO `ewc_level3_full` VALUES ('459', '56', '9', '10 12 09', '0', 'solid wastes from gas treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('460', '56', '10', '10 12 10', '1', 'solid wastes from gas treatment other than those mentioned in 10 12 09');
INSERT INTO `ewc_level3_full` VALUES ('461', '56', '11', '10 12 11', '0', 'wastes from glazing containing heavy metals');
INSERT INTO `ewc_level3_full` VALUES ('462', '56', '12', '10 12 12', '1', 'wastes from glazing other than those mentioned in 10 12 11');
INSERT INTO `ewc_level3_full` VALUES ('463', '56', '13', '10 12 13', '0', 'sludge from on-site effluent treatment');
INSERT INTO `ewc_level3_full` VALUES ('464', '56', '99', '10 12 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('465', '57', '1', '10 13 01', '0', 'waste preparation mixture before thermal processing');
INSERT INTO `ewc_level3_full` VALUES ('466', '57', '4', '10 13 04', '0', 'wastes from calcination and hydration of lime');
INSERT INTO `ewc_level3_full` VALUES ('467', '57', '6', '10 13 06', '0', 'particulates and dust (except 10 13 12 and 10 13 13)');
INSERT INTO `ewc_level3_full` VALUES ('468', '57', '7', '10 13 07', '0', 'sludges and filter cakes from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('469', '57', '9', '10 13 09', '0', 'wastes from asbestos-cement manufacture containing asbestos');
INSERT INTO `ewc_level3_full` VALUES ('470', '57', '10', '10 13 10', '1', 'wastes from asbestos-cement manufacture other than those mentioned in 10 13 09');
INSERT INTO `ewc_level3_full` VALUES ('471', '57', '11', '10 13 11', '0', 'wastes from cement-based composite materials other than those mentioned in 10 13 09 and 10 13 10');
INSERT INTO `ewc_level3_full` VALUES ('472', '57', '12', '10 13 12', '0', 'solid wastes from gas treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('473', '57', '13', '10 13 13', '1', 'solid wastes from gas treatment other than those mentioned in 10 13 12');
INSERT INTO `ewc_level3_full` VALUES ('474', '57', '14', '10 13 14', '0', 'waste concrete and concrete sludge');
INSERT INTO `ewc_level3_full` VALUES ('475', '57', '99', '10 13 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('476', '58', '1', '10 14 01', '0', 'waste from gas cleaning containing mercury');
INSERT INTO `ewc_level3_full` VALUES ('477', '59', '5', '11 01 05', '1', 'pickling acids');
INSERT INTO `ewc_level3_full` VALUES ('478', '59', '6', '11 01 06', '1', 'acids not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('479', '59', '7', '11 01 07', '1', 'pickling bases');
INSERT INTO `ewc_level3_full` VALUES ('480', '59', '8', '11 01 08', '1', 'phosphatising sludges');
INSERT INTO `ewc_level3_full` VALUES ('481', '59', '9', '11 01 09', '1', 'sludges and filter cakes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('482', '59', '10', '11 01 10', '1', 'sludges and filter cakes other than those mentioned in 11 01 09');
INSERT INTO `ewc_level3_full` VALUES ('483', '59', '11', '11 01 11', '0', 'aqueous rinsing liquids containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('484', '59', '12', '11 01 12', '1', 'aqueous rinsing liquids other than those mentioned in 11 01 11');
INSERT INTO `ewc_level3_full` VALUES ('485', '59', '13', '11 01 13', '0', 'degreasing wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('486', '59', '14', '11 01 14', '1', 'degreasing wastes other than those mentioned in 11 01 13');
INSERT INTO `ewc_level3_full` VALUES ('487', '59', '15', '11 01 15', '0', 'eluate and sludges from membrane systems or ion exchange systems containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('488', '59', '16', '11 01 16', '1', 'saturated or spent ion exchange resins');
INSERT INTO `ewc_level3_full` VALUES ('489', '59', '98', '11 01 98', '1', 'other wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('490', '59', '99', '11 01 99', '1', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('491', '60', '2', '11 02 02', '0', 'sludges from zinc hydrometallurgy (including jarosite, goethite)');
INSERT INTO `ewc_level3_full` VALUES ('492', '60', '3', '11 02 03', '1', 'wastes from the production of anodes for aqueous electrolytical processes');
INSERT INTO `ewc_level3_full` VALUES ('493', '60', '5', '11 02 05', '0', 'wastes from copper hydrometallurgical processes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('494', '60', '6', '11 02 06', '1', 'wastes from copper hydrometallurgical processes other than those mentioned in 11 02 05');
INSERT INTO `ewc_level3_full` VALUES ('495', '60', '7', '11 02 07', '0', 'other wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('496', '60', '99', '11 02 99', '1', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('497', '61', '1', '11 03 01', '0', 'wastes containing cyanide');
INSERT INTO `ewc_level3_full` VALUES ('498', '61', '2', '11 03 02', '1', 'other wastes');
INSERT INTO `ewc_level3_full` VALUES ('499', '62', '1', '11 05 01', '1', 'hard zinc');
INSERT INTO `ewc_level3_full` VALUES ('500', '62', '2', '11 05 02', '0', 'zinc ash');
INSERT INTO `ewc_level3_full` VALUES ('501', '62', '3', '11 05 03', '0', 'solid wastes from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('502', '62', '4', '11 05 04', '1', 'spent flux');
INSERT INTO `ewc_level3_full` VALUES ('503', '62', '99', '11 05 99', '1', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('504', '63', '1', '12 01 01', '0', 'ferrous metal filings and turnings');
INSERT INTO `ewc_level3_full` VALUES ('505', '63', '2', '12 01 02', '1', 'ferrous metal dust and particles');
INSERT INTO `ewc_level3_full` VALUES ('506', '63', '3', '12 01 03', '1', 'non-ferrous metal filings and turnings');
INSERT INTO `ewc_level3_full` VALUES ('507', '63', '4', '12 01 04', '1', 'non-ferrous metal dust and particles');
INSERT INTO `ewc_level3_full` VALUES ('508', '63', '5', '12 01 05', '1', 'plastics shavings and turnings');
INSERT INTO `ewc_level3_full` VALUES ('509', '63', '6', '12 01 06', '1', 'mineral-based machining oils containing halogens (except emulsions and solutions)');
INSERT INTO `ewc_level3_full` VALUES ('510', '63', '7', '12 01 07', '1', 'mineral-based machining oils free of halogens (except emulsions and solutions)');
INSERT INTO `ewc_level3_full` VALUES ('511', '63', '8', '12 01 08', '1', 'machining emulsions and solutions containing halogens');
INSERT INTO `ewc_level3_full` VALUES ('512', '63', '9', '12 01 09', '1', 'machining emulsions and solutions free of halogens');
INSERT INTO `ewc_level3_full` VALUES ('513', '63', '10', '12 01 10', '1', 'synthetic machining oils');
INSERT INTO `ewc_level3_full` VALUES ('514', '63', '12', '12 01 12', '1', 'spent waxes and fats');
INSERT INTO `ewc_level3_full` VALUES ('515', '63', '13', '12 01 13', '1', 'welding wastes');
INSERT INTO `ewc_level3_full` VALUES ('516', '63', '14', '12 01 14', '1', 'machining sludges containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('517', '63', '15', '12 01 15', '1', 'machining sludges other than those mentioned in 12 01 14');
INSERT INTO `ewc_level3_full` VALUES ('518', '63', '16', '12 01 16', '1', 'waste blasting material containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('519', '63', '17', '12 01 17', '1', 'waste blasting material other than those mentioned in 12 01 16');
INSERT INTO `ewc_level3_full` VALUES ('520', '63', '18', '12 01 18', '1', 'metal sludge (grinding, honing and lapping sludge) containing oil');
INSERT INTO `ewc_level3_full` VALUES ('521', '63', '19', '12 01 19', '1', 'readily biodegradable machining oil');
INSERT INTO `ewc_level3_full` VALUES ('522', '63', '20', '12 01 20', '1', 'spent grinding bodies and grinding materials containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('523', '63', '21', '12 01 21', '1', 'spent grinding bodies and grinding materials other than those mentioned in 12 01 20');
INSERT INTO `ewc_level3_full` VALUES ('524', '63', '99', '12 01 99', '1', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('525', '64', '1', '12 03 01', '1', 'aqueous washing liquids');
INSERT INTO `ewc_level3_full` VALUES ('526', '64', '2', '12 03 02', '1', 'steam degreasing wastes');
INSERT INTO `ewc_level3_full` VALUES ('527', '65', '1', '13 01 01', '1', 'hydraulic oils, containing PCBs (1)');
INSERT INTO `ewc_level3_full` VALUES ('528', '65', '4', '13 01 04', '1', 'chlorinated emulsions');
INSERT INTO `ewc_level3_full` VALUES ('529', '65', '5', '13 01 05', '1', 'non-chlorinated emulsions');
INSERT INTO `ewc_level3_full` VALUES ('530', '65', '9', '13 01 09', '1', 'mineral-based chlorinated hydraulic oils');
INSERT INTO `ewc_level3_full` VALUES ('531', '65', '10', '13 01 10', '1', 'mineral based non-chlorinated hydraulic oils');
INSERT INTO `ewc_level3_full` VALUES ('532', '65', '11', '13 01 11', '1', 'synthetic hydraulic oils');
INSERT INTO `ewc_level3_full` VALUES ('533', '65', '12', '13 01 12', '1', 'readily biodegradable hydraulic oils');
INSERT INTO `ewc_level3_full` VALUES ('534', '65', '13', '13 01 13', '1', 'other hydraulic oils');
INSERT INTO `ewc_level3_full` VALUES ('535', '66', '4', '13 02 04', '1', 'mineral-based chlorinated engine, gear and lubricating oils');
INSERT INTO `ewc_level3_full` VALUES ('536', '66', '5', '13 02 05', '1', 'mineral-based non-chlorinated engine, gear and lubricating oils');
INSERT INTO `ewc_level3_full` VALUES ('537', '66', '6', '13 02 06', '1', 'synthetic engine, gear and lubricating oils');
INSERT INTO `ewc_level3_full` VALUES ('538', '66', '7', '13 02 07', '1', 'readily biodegradable engine, gear and lubricating oils');
INSERT INTO `ewc_level3_full` VALUES ('539', '66', '8', '13 02 08', '1', 'other engine, gear and lubricating oils');
INSERT INTO `ewc_level3_full` VALUES ('540', '67', '1', '13 03 01', '1', 'insulating or heat transmission oils containing PCBs');
INSERT INTO `ewc_level3_full` VALUES ('541', '67', '6', '13 03 06', '1', 'mineral-based chlorinated insulating and heat transmission oils other than those mentioned in 13 03 01');
INSERT INTO `ewc_level3_full` VALUES ('542', '67', '7', '13 03 07', '1', 'mineral-based non-chlorinated insulating and heat transmission oils');
INSERT INTO `ewc_level3_full` VALUES ('543', '67', '8', '13 03 08', '1', 'synthetic insulating and heat transmission oils');
INSERT INTO `ewc_level3_full` VALUES ('544', '67', '9', '13 03 09', '1', 'readily biodegradable insulating and heat transmission oils');
INSERT INTO `ewc_level3_full` VALUES ('545', '67', '10', '13 03 10', '1', 'other insulating and heat transmission oils');
INSERT INTO `ewc_level3_full` VALUES ('546', '68', '1', '13 04 01', '0', 'bilge oils from inland navigation');
INSERT INTO `ewc_level3_full` VALUES ('547', '68', '2', '13 04 02', '0', 'bilge oils from jetty sewers');
INSERT INTO `ewc_level3_full` VALUES ('548', '68', '3', '13 04 03', '0', 'bilge oils from other navigation');
INSERT INTO `ewc_level3_full` VALUES ('549', '69', '1', '13 05 01', '0', 'solids from grit chambers and oil/water separators');
INSERT INTO `ewc_level3_full` VALUES ('550', '69', '2', '13 05 02', '0', 'sludges from oil/water separators');
INSERT INTO `ewc_level3_full` VALUES ('551', '69', '3', '13 05 03', '0', 'interceptor sludges');
INSERT INTO `ewc_level3_full` VALUES ('552', '69', '6', '13 05 06', '0', 'oil from oil/water separators');
INSERT INTO `ewc_level3_full` VALUES ('553', '69', '7', '13 05 07', '0', 'oily water from oil/water separators');
INSERT INTO `ewc_level3_full` VALUES ('554', '69', '8', '13 05 08', '1', 'mixtures of wastes from grit chambers and oil/water separators');
INSERT INTO `ewc_level3_full` VALUES ('555', '70', '1', '13 07 01', '1', 'fuel oil and diesel');
INSERT INTO `ewc_level3_full` VALUES ('556', '70', '2', '13 07 02', '1', 'petrol');
INSERT INTO `ewc_level3_full` VALUES ('557', '70', '3', '13 07 03', '0', 'other fuels (including mixtures)');
INSERT INTO `ewc_level3_full` VALUES ('558', '71', '1', '13 08 01', '0', 'desalter sludges or emulsions');
INSERT INTO `ewc_level3_full` VALUES ('559', '71', '2', '13 08 02', '0', 'other emulsions');
INSERT INTO `ewc_level3_full` VALUES ('560', '71', '99', '13 08 99', '1', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('561', '72', '1', '14 06 01', '1', 'chlorofluorocarbons, HCFC, HFC');
INSERT INTO `ewc_level3_full` VALUES ('562', '72', '2', '14 06 02', '1', 'other halogenated solvents and solvent mixtures');
INSERT INTO `ewc_level3_full` VALUES ('563', '72', '3', '14 06 03', '1', 'other solvents and solvent mixtures');
INSERT INTO `ewc_level3_full` VALUES ('564', '72', '4', '14 06 04', '1', 'sludges or solid wastes containing halogenated solvents');
INSERT INTO `ewc_level3_full` VALUES ('565', '72', '5', '14 06 05', '0', 'sludges or solid wastes containing other solvents');
INSERT INTO `ewc_level3_full` VALUES ('566', '73', '1', '15 01 01', '1', 'paper and cardboard packaging');
INSERT INTO `ewc_level3_full` VALUES ('567', '73', '2', '15 01 02', '1', 'plastic packaging');
INSERT INTO `ewc_level3_full` VALUES ('568', '73', '3', '15 01 03', '0', 'wooden packaging');
INSERT INTO `ewc_level3_full` VALUES ('569', '73', '4', '15 01 04', '0', 'metallic packaging');
INSERT INTO `ewc_level3_full` VALUES ('570', '73', '5', '15 01 05', '0', 'composite packaging');
INSERT INTO `ewc_level3_full` VALUES ('571', '73', '6', '15 01 06', '0', 'mixed packaging');
INSERT INTO `ewc_level3_full` VALUES ('572', '73', '7', '15 01 07', '0', 'glass packaging');
INSERT INTO `ewc_level3_full` VALUES ('573', '73', '9', '15 01 09', '0', 'textile packaging');
INSERT INTO `ewc_level3_full` VALUES ('574', '73', '10', '15 01 10', '1', 'packaging containing residues of or contaminated by dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('575', '73', '11', '15 01 11', '0', 'metallic packaging containing a dangerous solid porous matrix (for example asbestos), including empty pressure containers');
INSERT INTO `ewc_level3_full` VALUES ('576', '74', '2', '15 02 02', '0', 'absorbents, filter materials (including oil filters not otherwise specified), wiping cloths, protective clothing contaminated by dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('577', '74', '3', '15 02 03', '1', 'absorbents, filter materials, wiping cloths and protective clothing other than those mentioned in 15 02 02');
INSERT INTO `ewc_level3_full` VALUES ('578', '75', '3', '16 01 03', '1', 'end-of-life tyres');
INSERT INTO `ewc_level3_full` VALUES ('579', '75', '4', '16 01 04', '1', 'discarded vehicles');
INSERT INTO `ewc_level3_full` VALUES ('580', '75', '6', '16 01 06', '1', 'end-of-life vehicles, containing neither liquids nor other hazardous components');
INSERT INTO `ewc_level3_full` VALUES ('581', '75', '7', '16 01 07', '1', 'oil filters');
INSERT INTO `ewc_level3_full` VALUES ('582', '75', '8', '16 01 08', '0', 'components containing mercury');
INSERT INTO `ewc_level3_full` VALUES ('583', '75', '9', '16 01 09', '1', 'components containing PCBs');
INSERT INTO `ewc_level3_full` VALUES ('584', '75', '10', '16 01 10', '0', 'explosive components (for example air bags)');
INSERT INTO `ewc_level3_full` VALUES ('585', '75', '11', '16 01 11', '0', 'brake pads containing asbestos');
INSERT INTO `ewc_level3_full` VALUES ('586', '75', '12', '16 01 12', '1', 'brake pads other than those mentioned in 16 01 11');
INSERT INTO `ewc_level3_full` VALUES ('587', '75', '13', '16 01 13', '0', 'brake fluids');
INSERT INTO `ewc_level3_full` VALUES ('588', '75', '14', '16 01 14', '1', 'antifreeze fluids containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('589', '75', '15', '16 01 15', '0', 'antifreeze fluids other than those mentioned in 16 01 14');
INSERT INTO `ewc_level3_full` VALUES ('590', '75', '16', '16 01 16', '1', 'tanks for liquefied gas');
INSERT INTO `ewc_level3_full` VALUES ('591', '75', '17', '16 01 17', '1', 'ferrous metal');
INSERT INTO `ewc_level3_full` VALUES ('592', '75', '18', '16 01 18', '1', 'non-ferrous metal');
INSERT INTO `ewc_level3_full` VALUES ('593', '75', '19', '16 01 19', '1', 'plastic');
INSERT INTO `ewc_level3_full` VALUES ('594', '75', '20', '16 01 20', '0', 'glass');
INSERT INTO `ewc_level3_full` VALUES ('595', '75', '21', '16 01 21', '1', 'hazardous components other than those mentioned in 16 01 07 to 16 01 11 and 16 01 13 and 16 01 14');
INSERT INTO `ewc_level3_full` VALUES ('596', '75', '22', '16 01 22', '1', 'components not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('597', '75', '99', '16 01 99', '1', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('598', '76', '9', '16 02 09', '0', 'transformers and capacitors containing PCBs');
INSERT INTO `ewc_level3_full` VALUES ('599', '76', '10', '16 02 10', '1', 'discarded equipment containing or contaminated by PCBs other than those mentioned in 16 02 09');
INSERT INTO `ewc_level3_full` VALUES ('600', '76', '11', '16 02 11', '1', 'discarded equipment containing chlorofluorocarbons, HCFC, HFC');
INSERT INTO `ewc_level3_full` VALUES ('601', '76', '12', '16 02 12', '1', 'discarded equipment containing free asbestos');
INSERT INTO `ewc_level3_full` VALUES ('602', '76', '13', '16 02 13', '0', 'discarded equipment containing hazardous components (2) other than those mentioned in 16 02 09 to 16 02 12');
INSERT INTO `ewc_level3_full` VALUES ('603', '76', '14', '16 02 14', '0', 'discarded equipment other than those mentioned in 16 02 09 to 16 02 13');
INSERT INTO `ewc_level3_full` VALUES ('604', '76', '15', '16 02 15', '1', 'hazardous components removed from discarded equipment');
INSERT INTO `ewc_level3_full` VALUES ('605', '76', '16', '16 02 16', '1', 'components removed from discarded equipment other than those mentioned in 16 02 15');
INSERT INTO `ewc_level3_full` VALUES ('606', '77', '3', '16 03 03', '1', 'inorganic wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('607', '77', '4', '16 03 04', '0', 'inorganic wastes other than those mentioned in 16 03 03');
INSERT INTO `ewc_level3_full` VALUES ('608', '77', '5', '16 03 05', '0', 'organic wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('609', '77', '6', '16 03 06', '1', 'organic wastes other than those mentioned in 16 03 05');
INSERT INTO `ewc_level3_full` VALUES ('610', '77', '7', '16 03 07', '0', 'metallic mercury');
INSERT INTO `ewc_level3_full` VALUES ('611', '78', '1', '16 04 01', '0', 'waste ammunition');
INSERT INTO `ewc_level3_full` VALUES ('612', '78', '2', '16 04 02', '1', 'fireworks wastes');
INSERT INTO `ewc_level3_full` VALUES ('613', '78', '3', '16 04 03', '1', 'other waste explosives');
INSERT INTO `ewc_level3_full` VALUES ('614', '79', '4', '16 05 04', '1', 'gases in pressure containers (including halons) containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('615', '79', '5', '16 05 05', '1', 'gases in pressure containers other than those mentioned in 16 05 04');
INSERT INTO `ewc_level3_full` VALUES ('616', '79', '6', '16 05 06', '1', 'laboratory chemicals, consisting of or containing dangerous substances, including mixtures of laboratory chemicals');
INSERT INTO `ewc_level3_full` VALUES ('617', '79', '7', '16 05 07', '1', 'discarded inorganic chemicals consisting of or containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('618', '79', '8', '16 05 08', '1', 'discarded organic chemicals consisting of or containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('619', '79', '9', '16 05 09', '0', 'discarded chemicals other than those mentioned in 16 05 06, 16 05 07 or 16 05 08');
INSERT INTO `ewc_level3_full` VALUES ('620', '80', '1', '16 06 01', '1', 'lead batteries');
INSERT INTO `ewc_level3_full` VALUES ('621', '80', '2', '16 06 02', '0', 'Ni-Cd batteries');
INSERT INTO `ewc_level3_full` VALUES ('622', '80', '3', '16 06 03', '1', 'mercury-containing batteries');
INSERT INTO `ewc_level3_full` VALUES ('623', '80', '4', '16 06 04', '0', 'alkaline batteries (except 16 06 03)');
INSERT INTO `ewc_level3_full` VALUES ('624', '80', '5', '16 06 05', '1', 'other batteries and accumulators');
INSERT INTO `ewc_level3_full` VALUES ('625', '80', '6', '16 06 06', '0', 'separately collected electrolyte from batteries and accumulators');
INSERT INTO `ewc_level3_full` VALUES ('626', '81', '8', '16 07 08', '1', 'wastes containing oil');
INSERT INTO `ewc_level3_full` VALUES ('627', '81', '9', '16 07 09', '0', 'wastes containing other dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('628', '81', '99', '16 07 99', '1', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('629', '82', '1', '16 08 01', '0', 'spent catalysts containing gold, silver, rhenium, rhodium, palladium, iridium or platinum (except 16 08 07)');
INSERT INTO `ewc_level3_full` VALUES ('630', '82', '2', '16 08 02', '0', 'spent catalysts containing dangerous transition metals (3) or dangerous transition metal compounds');
INSERT INTO `ewc_level3_full` VALUES ('631', '82', '3', '16 08 03', '0', 'spent catalysts containing transition metals or transition metal compounds not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('632', '82', '4', '16 08 04', '0', 'spent fluid catalytic cracking catalysts (except 16 08 07)');
INSERT INTO `ewc_level3_full` VALUES ('633', '82', '5', '16 08 05', '1', 'spent catalysts containing phosphoric acid');
INSERT INTO `ewc_level3_full` VALUES ('634', '82', '6', '16 08 06', '0', 'spent liquids used as catalysts');
INSERT INTO `ewc_level3_full` VALUES ('635', '82', '7', '16 08 07', '0', 'spent catalysts contaminated with dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('636', '83', '1', '16 09 01', '0', 'permanganates, for example potassium permanganate');
INSERT INTO `ewc_level3_full` VALUES ('637', '83', '2', '16 09 02', '0', 'chromates, for example potassium chromate, potassium or sodium dichromate');
INSERT INTO `ewc_level3_full` VALUES ('638', '83', '3', '16 09 03', '1', 'peroxides, for example hydrogen peroxide');
INSERT INTO `ewc_level3_full` VALUES ('639', '83', '4', '16 09 04', '1', 'oxidising substances, not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('640', '84', '1', '16 10 01', '0', 'aqueous liquid wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('641', '84', '2', '16 10 02', '1', 'aqueous liquid wastes other than those mentioned in 16 10 01');
INSERT INTO `ewc_level3_full` VALUES ('642', '84', '3', '16 10 03', '0', 'aqueous concentrates containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('643', '84', '4', '16 10 04', '0', 'aqueous concentrates other than those mentioned in 16 10 03');
INSERT INTO `ewc_level3_full` VALUES ('644', '85', '1', '16 11 01', '0', 'carbon-based linings and refractories from metallurgical processes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('645', '85', '2', '16 11 02', '0', 'carbon-based linings and refractories from metallurgical processes others than those mentioned in 16 11 01,');
INSERT INTO `ewc_level3_full` VALUES ('646', '85', '3', '16 11 03', '0', 'other linings and refractories from metallurgical processes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('647', '85', '4', '16 11 04', '0', 'other linings and refractories from metallurgical processes other than those mentioned in 16 11 03');
INSERT INTO `ewc_level3_full` VALUES ('648', '85', '5', '16 11 05', '0', 'linings and refractories from non-metallurgical processes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('649', '85', '6', '16 11 06', '1', 'linings and refractories from non-metallurgical processes others than those mentioned in 16 11 05');
INSERT INTO `ewc_level3_full` VALUES ('650', '86', '1', '17 01 01', '1', 'concrete');
INSERT INTO `ewc_level3_full` VALUES ('651', '86', '2', '17 01 02', '0', 'bricks');
INSERT INTO `ewc_level3_full` VALUES ('652', '86', '3', '17 01 03', '0', 'tiles and ceramics');
INSERT INTO `ewc_level3_full` VALUES ('653', '86', '6', '17 01 06', '1', 'mixtures of, or separate fractions of concrete, bricks, tiles and ceramics containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('654', '86', '7', '17 01 07', '0', 'mixtures of concrete, bricks, tiles and ceramics other than those mentioned in 17 01 06');
INSERT INTO `ewc_level3_full` VALUES ('655', '87', '1', '17 02 01', '1', 'wood');
INSERT INTO `ewc_level3_full` VALUES ('656', '87', '2', '17 02 02', '0', 'glass');
INSERT INTO `ewc_level3_full` VALUES ('657', '87', '3', '17 02 03', '1', 'plastic');
INSERT INTO `ewc_level3_full` VALUES ('658', '87', '4', '17 02 04', '0', 'glass, plastic and wood containing or contaminated with dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('659', '88', '1', '17 03 01', '1', 'bituminous mixtures containing coal tar');
INSERT INTO `ewc_level3_full` VALUES ('660', '88', '2', '17 03 02', '1', 'bituminous mixtures other than those mentioned in 17 03 01');
INSERT INTO `ewc_level3_full` VALUES ('661', '88', '3', '17 03 03', '0', 'coal tar and tarred products');
INSERT INTO `ewc_level3_full` VALUES ('662', '89', '1', '17 04 01', '1', 'copper, bronze, brass');
INSERT INTO `ewc_level3_full` VALUES ('663', '89', '2', '17 04 02', '1', 'aluminium');
INSERT INTO `ewc_level3_full` VALUES ('664', '89', '3', '17 04 03', '0', 'lead');
INSERT INTO `ewc_level3_full` VALUES ('665', '89', '4', '17 04 04', '1', 'zinc');
INSERT INTO `ewc_level3_full` VALUES ('666', '89', '5', '17 04 05', '1', 'iron and steel');
INSERT INTO `ewc_level3_full` VALUES ('667', '89', '6', '17 04 06', '1', 'tin');
INSERT INTO `ewc_level3_full` VALUES ('668', '89', '7', '17 04 07', '0', 'mixed metals');
INSERT INTO `ewc_level3_full` VALUES ('669', '89', '9', '17 04 09', '0', 'metal waste contaminated with dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('670', '89', '10', '17 04 10', '0', 'cables containing oil, coal tar and other dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('671', '89', '11', '17 04 11', '1', 'cables other than those mentioned in 17 04 10');
INSERT INTO `ewc_level3_full` VALUES ('672', '90', '3', '17 05 03', '0', 'soil and stones containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('673', '90', '4', '17 05 04', '1', 'soil and stones other than those mentioned in 17 05 03');
INSERT INTO `ewc_level3_full` VALUES ('674', '90', '5', '17 05 05', '0', 'dredging spoil containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('675', '90', '6', '17 05 06', '1', 'dredging spoil other than those mentioned in 17 05 05');
INSERT INTO `ewc_level3_full` VALUES ('676', '90', '7', '17 05 07', '0', 'track ballast containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('677', '90', '8', '17 05 08', '1', 'track ballast other than those mentioned in 17 05 07');
INSERT INTO `ewc_level3_full` VALUES ('678', '91', '1', '17 06 01', '0', 'insulation materials containing asbestos');
INSERT INTO `ewc_level3_full` VALUES ('679', '91', '3', '17 06 03', '1', 'other insulation materials consisting of or containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('680', '91', '4', '17 06 04', '0', 'insulation materials other than those mentioned in 17 06 01 and 17 06 03');
INSERT INTO `ewc_level3_full` VALUES ('681', '91', '5', '17 06 05', '1', 'construction materials containing asbestos');
INSERT INTO `ewc_level3_full` VALUES ('682', '92', '1', '17 08 01', '0', 'gypsum-based construction materials contaminated with dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('683', '92', '2', '17 08 02', '1', 'gypsum-based construction materials other than those mentioned in 17 08 01');
INSERT INTO `ewc_level3_full` VALUES ('684', '93', '1', '17 09 01', '0', 'construction and demolition wastes containing mercury');
INSERT INTO `ewc_level3_full` VALUES ('685', '93', '2', '17 09 02', '0', 'construction and demolition wastes containing PCB (for example PCB-containing sealants, PCB-containing resin-based floorings, PCB-containing sealed glazing units, PCB-containing capacitors)');
INSERT INTO `ewc_level3_full` VALUES ('686', '93', '3', '17 09 03', '1', 'other construction and demolition wastes (including mixed wastes) containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('687', '93', '4', '17 09 04', '1', 'mixed construction and demolition wastes other than those mentioned in 17 09 01, 17 09 02 and 17 09 03');
INSERT INTO `ewc_level3_full` VALUES ('688', '94', '1', '18 01 01', '1', 'sharps (except 18 01 03)');
INSERT INTO `ewc_level3_full` VALUES ('689', '94', '2', '18 01 02', '1', 'body parts and organs including blood bags and blood preserves (except 18 01 03)');
INSERT INTO `ewc_level3_full` VALUES ('690', '94', '3', '18 01 03', '1', 'wastes whose collection and disposal is subject to special requirements in order to prevent infection');
INSERT INTO `ewc_level3_full` VALUES ('691', '94', '4', '18 01 04', '0', 'wastes whose collection and disposal is not subject to special requirements in order to prevent infection (for example dressings, plaster casts, linen, disposable clothing, diapers)');
INSERT INTO `ewc_level3_full` VALUES ('692', '94', '6', '18 01 06', '1', 'chemicals consisting of or containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('693', '94', '7', '18 01 07', '0', 'chemicals other than those mentioned in 18 01 06');
INSERT INTO `ewc_level3_full` VALUES ('694', '94', '8', '18 01 08', '1', 'cytotoxic and cytostatic medicines');
INSERT INTO `ewc_level3_full` VALUES ('695', '94', '9', '18 01 09', '0', 'medicines other than those mentioned in 18 01 08');
INSERT INTO `ewc_level3_full` VALUES ('696', '94', '10', '18 01 10', '1', 'amalgam waste from dental care');
INSERT INTO `ewc_level3_full` VALUES ('697', '95', '1', '18 02 01', '0', 'sharps (except 18 02 02)');
INSERT INTO `ewc_level3_full` VALUES ('698', '95', '2', '18 02 02', '0', 'wastes whose collection and disposal is subject to special requirements in order to prevent infection');
INSERT INTO `ewc_level3_full` VALUES ('699', '95', '3', '18 02 03', '0', 'wastes whose collection and disposal is not subject to special requirements in order to prevent infection');
INSERT INTO `ewc_level3_full` VALUES ('700', '95', '5', '18 02 05', '0', 'chemicals consisting of or containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('701', '95', '6', '18 02 06', '1', 'chemicals other than those mentioned in 18 02 05');
INSERT INTO `ewc_level3_full` VALUES ('702', '95', '7', '18 02 07', '1', 'cytotoxic and cytostatic medicines');
INSERT INTO `ewc_level3_full` VALUES ('703', '95', '8', '18 02 08', '0', 'medicines other than those mentioned in 18 02 07');
INSERT INTO `ewc_level3_full` VALUES ('704', '96', '2', '19 01 02', '1', 'ferrous materials removed from bottom ash');
INSERT INTO `ewc_level3_full` VALUES ('705', '96', '5', '19 01 05', '1', 'filter cake from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('706', '96', '6', '19 01 06', '1', 'aqueous liquid wastes from gas treatment and other aqueous liquid wastes');
INSERT INTO `ewc_level3_full` VALUES ('707', '96', '7', '19 01 07', '0', 'solid wastes from gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('708', '96', '10', '19 01 10', '1', 'spent activated carbon from flue-gas treatment');
INSERT INTO `ewc_level3_full` VALUES ('709', '96', '11', '19 01 11', '0', 'bottom ash and slag containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('710', '96', '12', '19 01 12', '1', 'bottom ash and slag other than those mentioned in 19 01 11');
INSERT INTO `ewc_level3_full` VALUES ('711', '96', '13', '19 01 13', '0', 'fly ash containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('712', '96', '14', '19 01 14', '1', 'fly ash other than those mentioned in 19 01 13');
INSERT INTO `ewc_level3_full` VALUES ('713', '96', '15', '19 01 15', '0', 'boiler dust containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('714', '96', '16', '19 01 16', '0', 'boiler dust other than those mentioned in 19 01 15');
INSERT INTO `ewc_level3_full` VALUES ('715', '96', '17', '19 01 17', '1', 'pyrolysis wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('716', '96', '18', '19 01 18', '1', 'pyrolysis wastes other than those mentioned in 19 01 17');
INSERT INTO `ewc_level3_full` VALUES ('717', '96', '19', '19 01 19', '0', 'sands from fluidised beds');
INSERT INTO `ewc_level3_full` VALUES ('718', '96', '99', '19 01 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('719', '97', '3', '19 02 03', '0', 'premixed wastes composed only of non-hazardous wastes');
INSERT INTO `ewc_level3_full` VALUES ('720', '97', '4', '19 02 04', '0', 'premixed wastes composed of at least one hazardous waste');
INSERT INTO `ewc_level3_full` VALUES ('721', '97', '5', '19 02 05', '0', 'sludges from physico/chemical treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('722', '97', '6', '19 02 06', '0', 'sludges from physico/chemical treatment other than those mentioned in 19 02 05');
INSERT INTO `ewc_level3_full` VALUES ('723', '97', '7', '19 02 07', '0', 'oil and concentrates from separation');
INSERT INTO `ewc_level3_full` VALUES ('724', '97', '8', '19 02 08', '0', 'liquid combustible wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('725', '97', '9', '19 02 09', '0', 'solid combustible wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('726', '97', '10', '19 02 10', '0', 'combustible wastes other than those mentioned in 19 02 08 and 19 02 09');
INSERT INTO `ewc_level3_full` VALUES ('727', '97', '11', '19 02 11', '1', 'other wastes containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('728', '97', '99', '19 02 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('729', '98', '4', '19 03 04', '0', 'wastes marked as hazardous, partly (5) stabilised');
INSERT INTO `ewc_level3_full` VALUES ('730', '98', '5', '19 03 05', '0', 'stabilised wastes other than those mentioned in 19 03 04');
INSERT INTO `ewc_level3_full` VALUES ('731', '98', '6', '19 03 06', '0', 'wastes marked as hazardous, solidified');
INSERT INTO `ewc_level3_full` VALUES ('732', '98', '7', '19 03 07', '1', 'solidified wastes other than those mentioned in 19 03 06');
INSERT INTO `ewc_level3_full` VALUES ('733', '99', '1', '19 04 01', '1', 'vitrified waste');
INSERT INTO `ewc_level3_full` VALUES ('734', '99', '2', '19 04 02', '1', 'fly ash and other flue-gas treatment wastes');
INSERT INTO `ewc_level3_full` VALUES ('735', '99', '3', '19 04 03', '0', 'non-vitrified solid phase');
INSERT INTO `ewc_level3_full` VALUES ('736', '99', '4', '19 04 04', '1', 'aqueous liquid wastes from vitrified waste tempering');
INSERT INTO `ewc_level3_full` VALUES ('737', '100', '1', '19 05 01', '1', 'non-composted fraction of municipal and similar wastes');
INSERT INTO `ewc_level3_full` VALUES ('738', '100', '2', '19 05 02', '0', 'non-composted fraction of animal and vegetable waste');
INSERT INTO `ewc_level3_full` VALUES ('739', '100', '3', '19 05 03', '1', 'off-specification compost');
INSERT INTO `ewc_level3_full` VALUES ('740', '100', '99', '19 05 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('741', '101', '3', '19 06 03', '0', 'liquor from anaerobic treatment of municipal waste');
INSERT INTO `ewc_level3_full` VALUES ('742', '101', '4', '19 06 04', '0', 'digestate from anaerobic treatment of municipal waste');
INSERT INTO `ewc_level3_full` VALUES ('743', '101', '5', '19 06 05', '0', 'liquor from anaerobic treatment of animal and vegetable waste');
INSERT INTO `ewc_level3_full` VALUES ('744', '101', '6', '19 06 06', '0', 'digestate from anaerobic treatment of animal and vegetable waste');
INSERT INTO `ewc_level3_full` VALUES ('745', '101', '99', '19 06 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('746', '102', '2', '19 07 02', '0', 'landfill leachate containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('747', '102', '3', '19 07 03', '0', 'landfill leachate other than those mentioned in 19 07 02');
INSERT INTO `ewc_level3_full` VALUES ('748', '103', '1', '19 08 01', '0', 'screenings');
INSERT INTO `ewc_level3_full` VALUES ('749', '103', '2', '19 08 02', '0', 'waste from desanding');
INSERT INTO `ewc_level3_full` VALUES ('750', '103', '5', '19 08 05', '0', 'sludges from treatment of urban waste water');
INSERT INTO `ewc_level3_full` VALUES ('751', '103', '6', '19 08 06', '1', 'saturated or spent ion exchange resins');
INSERT INTO `ewc_level3_full` VALUES ('752', '103', '7', '19 08 07', '0', 'solutions and sludges from regeneration of ion exchangers');
INSERT INTO `ewc_level3_full` VALUES ('753', '103', '8', '19 08 08', '1', 'membrane system waste containing heavy metals');
INSERT INTO `ewc_level3_full` VALUES ('754', '103', '9', '19 08 09', '0', 'grease and oil mixture from oil/water separation containing edible oil and fats');
INSERT INTO `ewc_level3_full` VALUES ('755', '103', '10', '19 08 10', '1', 'grease and oil mixture from oil/water separation other than those mentioned in 19 08 09');
INSERT INTO `ewc_level3_full` VALUES ('756', '103', '11', '19 08 11', '1', 'sludges containing dangerous substances from biological treatment of industrial waste water');
INSERT INTO `ewc_level3_full` VALUES ('757', '103', '12', '19 08 12', '1', 'sludges from biological treatment of industrial waste water other than those mentioned in 19 08 11');
INSERT INTO `ewc_level3_full` VALUES ('758', '103', '13', '19 08 13', '1', 'sludges containing dangerous substances from other treatment of industrial waste water');
INSERT INTO `ewc_level3_full` VALUES ('759', '103', '14', '19 08 14', '1', 'sludges from other treatment of industrial waste water other than those mentioned in 19 08 13');
INSERT INTO `ewc_level3_full` VALUES ('760', '103', '99', '19 08 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('761', '104', '1', '19 09 01', '1', 'solid waste from primary filtration and screenings');
INSERT INTO `ewc_level3_full` VALUES ('762', '104', '2', '19 09 02', '0', 'sludges from water clarification');
INSERT INTO `ewc_level3_full` VALUES ('763', '104', '3', '19 09 03', '0', 'sludges from decarbonation');
INSERT INTO `ewc_level3_full` VALUES ('764', '104', '4', '19 09 04', '0', 'spent activated carbon');
INSERT INTO `ewc_level3_full` VALUES ('765', '104', '5', '19 09 05', '0', 'saturated or spent ion exchange resins');
INSERT INTO `ewc_level3_full` VALUES ('766', '104', '6', '19 09 06', '0', 'solutions and sludges from regeneration of ion exchangers');
INSERT INTO `ewc_level3_full` VALUES ('767', '104', '99', '19 09 99', '0', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('768', '105', '1', '19 10 01', '1', 'iron and steel waste');
INSERT INTO `ewc_level3_full` VALUES ('769', '105', '2', '19 10 02', '0', 'non-ferrous waste');
INSERT INTO `ewc_level3_full` VALUES ('770', '105', '3', '19 10 03', '0', 'fluff-light fraction and dust containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('771', '105', '4', '19 10 04', '0', 'fluff-light fraction and dust other than those mentioned in 19 10 03');
INSERT INTO `ewc_level3_full` VALUES ('772', '105', '5', '19 10 05', '0', 'other fractions containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('773', '105', '6', '19 10 06', '1', 'other fractions other than those mentioned in 19 10 05');
INSERT INTO `ewc_level3_full` VALUES ('774', '106', '1', '19 11 01', '0', 'spent filter clays');
INSERT INTO `ewc_level3_full` VALUES ('775', '106', '2', '19 11 02', '1', 'acid tars');
INSERT INTO `ewc_level3_full` VALUES ('776', '106', '3', '19 11 03', '0', 'aqueous liquid wastes');
INSERT INTO `ewc_level3_full` VALUES ('777', '106', '4', '19 11 04', '1', 'wastes from cleaning of fuel with bases');
INSERT INTO `ewc_level3_full` VALUES ('778', '106', '5', '19 11 05', '0', 'sludges from on-site effluent treatment containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('779', '106', '6', '19 11 06', '1', 'sludges from on-site effluent treatment other than those mentioned in 19 11 05');
INSERT INTO `ewc_level3_full` VALUES ('780', '106', '7', '19 11 07', '0', 'wastes from flue-gas cleaning');
INSERT INTO `ewc_level3_full` VALUES ('781', '106', '99', '19 11 99', '1', 'wastes not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('782', '107', '1', '19 12 01', '0', 'paper and cardboard');
INSERT INTO `ewc_level3_full` VALUES ('783', '107', '2', '19 12 02', '0', 'ferrous metal');
INSERT INTO `ewc_level3_full` VALUES ('784', '107', '3', '19 12 03', '0', 'non-ferrous metal');
INSERT INTO `ewc_level3_full` VALUES ('785', '107', '4', '19 12 04', '0', 'plastic and rubber');
INSERT INTO `ewc_level3_full` VALUES ('786', '107', '5', '19 12 05', '0', 'glass');
INSERT INTO `ewc_level3_full` VALUES ('787', '107', '6', '19 12 06', '0', 'wood containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('788', '107', '7', '19 12 07', '1', 'wood other than that mentioned in 19 12 06');
INSERT INTO `ewc_level3_full` VALUES ('789', '107', '8', '19 12 08', '1', 'textiles');
INSERT INTO `ewc_level3_full` VALUES ('790', '107', '9', '19 12 09', '1', 'minerals (for example sand, stones)');
INSERT INTO `ewc_level3_full` VALUES ('791', '107', '10', '19 12 10', '1', 'combustible waste (refuse derived fuel)');
INSERT INTO `ewc_level3_full` VALUES ('792', '107', '11', '19 12 11', '1', 'other wastes (including mixtures of materials) from mechanical treatment of waste containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('793', '107', '12', '19 12 12', '1', 'other wastes (including mixtures of materials) from mechanical treatment of wastes other than those mentioned in 19 12 11');
INSERT INTO `ewc_level3_full` VALUES ('794', '108', '1', '19 13 01', '1', 'solid wastes from soil remediation containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('795', '108', '2', '19 13 02', '0', 'solid wastes from soil remediation other than those mentioned in 19 13 01');
INSERT INTO `ewc_level3_full` VALUES ('796', '108', '3', '19 13 03', '1', 'sludges from soil remediation containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('797', '108', '4', '19 13 04', '1', 'sludges from soil remediation other than those mentioned in 19 13 03');
INSERT INTO `ewc_level3_full` VALUES ('798', '108', '5', '19 13 05', '0', 'sludges from groundwater remediation containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('799', '108', '6', '19 13 06', '1', 'sludges from groundwater remediation other than those mentioned in 19 13 05');
INSERT INTO `ewc_level3_full` VALUES ('800', '108', '7', '19 13 07', '0', 'aqueous liquid wastes and aqueous concentrates from groundwater remediation containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('801', '108', '8', '19 13 08', '1', 'aqueous liquid wastes and aqueous concentrates from groundwater remediation other than those mentioned in 19 13 07');
INSERT INTO `ewc_level3_full` VALUES ('802', '109', '1', '20 01 01', '0', 'paper and cardboard');
INSERT INTO `ewc_level3_full` VALUES ('803', '109', '2', '20 01 02', '1', 'glass');
INSERT INTO `ewc_level3_full` VALUES ('804', '109', '8', '20 01 08', '0', 'biodegradable kitchen and canteen waste');
INSERT INTO `ewc_level3_full` VALUES ('805', '109', '10', '20 01 10', '1', 'clothes');
INSERT INTO `ewc_level3_full` VALUES ('806', '109', '11', '20 01 11', '0', 'textiles');
INSERT INTO `ewc_level3_full` VALUES ('807', '109', '13', '20 01 13', '1', 'solvents');
INSERT INTO `ewc_level3_full` VALUES ('808', '109', '14', '20 01 14', '0', 'acids');
INSERT INTO `ewc_level3_full` VALUES ('809', '109', '15', '20 01 15', '0', 'alkalines');
INSERT INTO `ewc_level3_full` VALUES ('810', '109', '17', '20 01 17', '0', 'photochemicals');
INSERT INTO `ewc_level3_full` VALUES ('811', '109', '19', '20 01 19', '0', 'pesticides');
INSERT INTO `ewc_level3_full` VALUES ('812', '109', '21', '20 01 21', '0', 'fluorescent tubes and other mercury-containing waste');
INSERT INTO `ewc_level3_full` VALUES ('813', '109', '23', '20 01 23', '0', 'discarded equipment containing chlorofluorocarbons');
INSERT INTO `ewc_level3_full` VALUES ('814', '109', '25', '20 01 25', '0', 'edible oil and fat');
INSERT INTO `ewc_level3_full` VALUES ('815', '109', '26', '20 01 26', '0', 'oil and fat other than those mentioned in 20 01 25');
INSERT INTO `ewc_level3_full` VALUES ('816', '109', '27', '20 01 27', '0', 'paint, inks, adhesives and resins containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('817', '109', '28', '20 01 28', '0', 'paint, inks, adhesives and resins other than those mentioned in 20 01 27');
INSERT INTO `ewc_level3_full` VALUES ('818', '109', '29', '20 01 29', '0', 'detergents containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('819', '109', '30', '20 01 30', '0', 'detergents other than those mentioned in 20 01 29');
INSERT INTO `ewc_level3_full` VALUES ('820', '109', '31', '20 01 31', '0', 'cytotoxic and cytostatic medicines');
INSERT INTO `ewc_level3_full` VALUES ('821', '109', '32', '20 01 32', '0', 'medicines other than those mentioned in 20 01 31');
INSERT INTO `ewc_level3_full` VALUES ('822', '109', '33', '20 01 33', '0', 'batteries and accumulators included in 16 06 01, 16 06 02 or 16 06 03 and unsorted batteries and accumulators containing these batteries');
INSERT INTO `ewc_level3_full` VALUES ('823', '109', '34', '20 01 34', '1', 'batteries and accumulators other than those mentioned in 20 01 33');
INSERT INTO `ewc_level3_full` VALUES ('824', '109', '35', '20 01 35', '0', 'discarded electrical and electronic equipment other than those mentioned in 20 01 21 and 20 01 23 containing hazardous components (6)');
INSERT INTO `ewc_level3_full` VALUES ('825', '109', '36', '20 01 36', '1', 'discarded electrical and electronic equipment other than those mentioned in 20 01 21, 20 01 23 and 20 01 35');
INSERT INTO `ewc_level3_full` VALUES ('826', '109', '37', '20 01 37', '0', 'wood containing dangerous substances');
INSERT INTO `ewc_level3_full` VALUES ('827', '109', '38', '20 01 38', '1', 'wood other than that mentioned in 20 01 37');
INSERT INTO `ewc_level3_full` VALUES ('828', '109', '39', '20 01 39', '0', 'plastics');
INSERT INTO `ewc_level3_full` VALUES ('829', '109', '40', '20 01 40', '0', 'metals');
INSERT INTO `ewc_level3_full` VALUES ('830', '109', '41', '20 01 41', '0', 'wastes from chimney sweeping');
INSERT INTO `ewc_level3_full` VALUES ('831', '109', '99', '20 01 99', '0', 'other fractions not otherwise specified');
INSERT INTO `ewc_level3_full` VALUES ('832', '110', '1', '20 02 01', '0', 'biodegradable waste');
INSERT INTO `ewc_level3_full` VALUES ('833', '110', '2', '20 02 02', '0', 'soil and stones');
INSERT INTO `ewc_level3_full` VALUES ('834', '110', '3', '20 02 03', '0', 'other non-biodegradable wastes');
INSERT INTO `ewc_level3_full` VALUES ('835', '111', '1', '20 03 01', '0', 'mixed municipal waste');
INSERT INTO `ewc_level3_full` VALUES ('836', '111', '2', '20 03 02', '0', 'waste from markets');
INSERT INTO `ewc_level3_full` VALUES ('837', '111', '3', '20 03 03', '0', 'street-cleaning residues');
INSERT INTO `ewc_level3_full` VALUES ('838', '111', '4', '20 03 04', '0', 'septic tank sludge');
INSERT INTO `ewc_level3_full` VALUES ('839', '111', '6', '20 03 06', '0', 'waste from sewage cleaning');
INSERT INTO `ewc_level3_full` VALUES ('840', '111', '7', '20 03 07', '0', 'bulky waste');
INSERT INTO `ewc_level3_full` VALUES ('841', '111', '99', '20 03 99', '0', 'municipal wastes not otherwise specified');

-- ----------------------------
-- Table structure for ewc_level3_subset19
-- ----------------------------
DROP TABLE IF EXISTS `ewc_level3_subset19`;
CREATE TABLE `ewc_level3_subset19` (
  `id` varchar(255) DEFAULT NULL,
  `parent_id` varchar(255) DEFAULT NULL,
  `EWC_id` varchar(255) DEFAULT NULL,
  `EWC_level3` varchar(255) DEFAULT NULL,
  `hazardous` varchar(255) DEFAULT NULL,
  `description` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of ewc_level3_subset19
-- ----------------------------
INSERT INTO `ewc_level3_subset19` VALUES ('12', '3', '8', '01 04 08', '0', 'waste gravel and crushed rocks other than those mentioned in 01 04 07');
INSERT INTO `ewc_level3_subset19` VALUES ('37', '6', '3', '02 02 03', '0', 'materials unsuitable for consumption or processing');
INSERT INTO `ewc_level3_subset19` VALUES ('106', '17', '5', '05 01 05', '1', 'oil spills');
INSERT INTO `ewc_level3_subset19` VALUES ('131', '20', '5', '06 01 05', '1', 'nitric acid and nitrous acid');
INSERT INTO `ewc_level3_subset19` VALUES ('137', '21', '5', '06 02 05', '1', 'other bases');
INSERT INTO `ewc_level3_subset19` VALUES ('265', '40', '1', '08 02 01', '0', 'waste coating powders');
INSERT INTO `ewc_level3_subset19` VALUES ('297', '44', '7', '09 01 07', '0', 'photographic film and paper containing silver or silver compounds');
INSERT INTO `ewc_level3_subset19` VALUES ('298', '44', '8', '09 01 08', '0', 'photographic film and paper free of silver or silver compounds');
INSERT INTO `ewc_level3_subset19` VALUES ('506', '63', '3', '12 01 03', '1', 'non-ferrous metal filings and turnings');
INSERT INTO `ewc_level3_subset19` VALUES ('508', '63', '5', '12 01 05', '1', 'plastics shavings and turnings');
INSERT INTO `ewc_level3_subset19` VALUES ('567', '73', '2', '15 01 02', '1', 'plastic packaging');
INSERT INTO `ewc_level3_subset19` VALUES ('569', '73', '4', '15 01 04', '0', 'metallic packaging');
INSERT INTO `ewc_level3_subset19` VALUES ('594', '75', '20', '16 01 20', '0', 'glass');
INSERT INTO `ewc_level3_subset19` VALUES ('664', '89', '3', '17 04 03', '0', 'lead');
INSERT INTO `ewc_level3_subset19` VALUES ('666', '89', '5', '17 04 05', '1', 'iron and steel');
INSERT INTO `ewc_level3_subset19` VALUES ('683', '92', '2', '17 08 02', '1', 'gypsum-based construction materials other than those mentioned in 17 08 01');
INSERT INTO `ewc_level3_subset19` VALUES ('771', '105', '4', '19 10 04', '0', 'fluff-light fraction and dust other than those mentioned in 19 10 03');
INSERT INTO `ewc_level3_subset19` VALUES ('782', '107', '1', '19 12 01', '0', 'paper and cardboard');
INSERT INTO `ewc_level3_subset19` VALUES ('833', '110', '2', '20 02 02', '0', 'soil and stones');

-- ----------------------------
-- Table structure for ewc_level3_subset5
-- ----------------------------
DROP TABLE IF EXISTS `ewc_level3_subset5`;
CREATE TABLE `ewc_level3_subset5` (
  `id` varchar(255) DEFAULT NULL,
  `parent_id` varchar(255) DEFAULT NULL,
  `EWC_id` varchar(255) DEFAULT NULL,
  `EWC_level3` varchar(255) DEFAULT NULL,
  `hazardous` varchar(255) DEFAULT NULL,
  `description` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of ewc_level3_subset5
-- ----------------------------
INSERT INTO `ewc_level3_subset5` VALUES ('566', '73', '1', '15 01 01', '1', 'paper and cardboard packaging');
INSERT INTO `ewc_level3_subset5` VALUES ('567', '73', '2', '15 01 02', '1', 'plastic packaging');
INSERT INTO `ewc_level3_subset5` VALUES ('590', '75', '16', '16 01 16', '1', 'tanks for liquefied gas');
INSERT INTO `ewc_level3_subset5` VALUES ('591', '75', '17', '16 01 17', '1', 'ferrous metal');
INSERT INTO `ewc_level3_subset5` VALUES ('506', '63', '3', '12 01 03', '1', 'non-ferrous metal filings and turnings');

-- ----------------------------
-- Table structure for ewc_level3_subset7
-- ----------------------------
DROP TABLE IF EXISTS `ewc_level3_subset7`;
CREATE TABLE `ewc_level3_subset7` (
  `id` varchar(255) DEFAULT NULL,
  `parent_id` varchar(255) DEFAULT NULL,
  `EWC_id` varchar(255) DEFAULT NULL,
  `EWC_level3` varchar(255) DEFAULT NULL,
  `hazardous` varchar(255) DEFAULT NULL,
  `description` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of ewc_level3_subset7
-- ----------------------------
INSERT INTO `ewc_level3_subset7` VALUES ('106', '17', '5', '05 01 05', '1', 'oil spills');
INSERT INTO `ewc_level3_subset7` VALUES ('265', '40', '1', '08 02 01', '0', 'waste coating powders');
INSERT INTO `ewc_level3_subset7` VALUES ('508', '63', '5', '12 01 05', '1', 'plastics shavings and turnings');
INSERT INTO `ewc_level3_subset7` VALUES ('567', '73', '2', '15 01 02', '1', 'plastic packaging');
INSERT INTO `ewc_level3_subset7` VALUES ('569', '73', '4', '15 01 04', '0', 'metallic packaging');
INSERT INTO `ewc_level3_subset7` VALUES ('594', '75', '20', '16 01 20', '0', 'glass');
INSERT INTO `ewc_level3_subset7` VALUES ('833', '110', '2', '20 02 02', '0', 'soil and stones');

-- ----------------------------
-- Table structure for workshop_items2
-- ----------------------------
DROP TABLE IF EXISTS `workshop_items2`;
CREATE TABLE `workshop_items2` (
  `id` bigint(2) NOT NULL AUTO_INCREMENT,
  `Cluster` varchar(255) DEFAULT NULL,
  `Workshop` varchar(2) DEFAULT NULL,
  `Company_ID` bigint(20) DEFAULT NULL,
  `Waste_description` text,
  `waste_description_original` varchar(2048) DEFAULT NULL,
  `language` varchar(20) DEFAULT NULL,
  `Type` varchar(255) DEFAULT NULL,
  `Wastecode` varchar(255) DEFAULT NULL,
  `Have_want` varchar(255) DEFAULT NULL,
  `Quantity` bigint(20) DEFAULT NULL,
  `Measure_unit` varchar(255) DEFAULT NULL,
  `Frequency` varchar(255) DEFAULT NULL,
  `Remarks` text,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=35 DEFAULT CHARSET=latin1;

-- ----------------------------
-- Records of workshop_items2
-- ----------------------------
INSERT INTO `workshop_items2` VALUES ('34', 'Turkey', 'A', '4', 'Iron and steel scrap', 'Iron and steel scrap', 'EN', 'Material', '16 01 17', 'Have', '40', 'tons', 'year', 'An alternative fuel for use in cement production. Can take 1500-2000 tons / year capacity.');

-- ----------------------------
-- Table structure for workshop_items2_full
-- ----------------------------
DROP TABLE IF EXISTS `workshop_items2_full`;
CREATE TABLE `workshop_items2_full` (
  `id` bigint(2) NOT NULL AUTO_INCREMENT,
  `Cluster` varchar(255) DEFAULT NULL,
  `Workshop` varchar(2) DEFAULT NULL,
  `Company_ID` bigint(20) DEFAULT NULL,
  `Waste_description` text,
  `waste_description_original` varchar(2048) DEFAULT NULL,
  `language` varchar(20) DEFAULT NULL,
  `Type` varchar(255) DEFAULT NULL,
  `Wastecode` varchar(255) DEFAULT NULL,
  `Have_want` varchar(255) DEFAULT NULL,
  `Quantity` bigint(20) DEFAULT NULL,
  `Measure_unit` varchar(255) DEFAULT NULL,
  `Frequency` varchar(255) DEFAULT NULL,
  `Remarks` text,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=750 DEFAULT CHARSET=latin1;

-- ----------------------------
-- Records of workshop_items2_full
-- ----------------------------
INSERT INTO `workshop_items2_full` VALUES ('1', 'Turkey', 'A', '1', 'Lab: Solid waste with the relevant laboratories in existing GC-MS elemental analysis, calorimetry , oxygen permeability , extruders, including laboratory analysis on the FCP- MS instrument can be analyzed to support research projects . ( A part of the Lab is accredited )', 'Lab: Solid waste with the relevant laboratories in existing GC-MS elemental analysis, calorimetry , oxygen permeability , extruders, including laboratory analysis on the FCP- MS instrument can be analyzed to support research projects . ( A part of the Lab is accredited )', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Safyem A.?.: capable to dried fodder;\nAÜ: Producing biopolymer from orange peel, the company can contact with packing agents;');
INSERT INTO `workshop_items2_full` VALUES ('2', 'Turkey', 'A', '1', 'Waste and hazardous waste management: Waste and hazardous waste management, management of AEEE , sludge decision support systems , life cycle analysis, etc. It can be carried out joint projects with companies on issues .', 'Waste and hazardous waste management: Waste and hazardous waste management, management of AEEE , sludge decision support systems , life cycle analysis, etc. It can be carried out joint projects with companies on issues .', 'EN', 'Service', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('3', 'Turkey', 'A', '2', 'plastic chips: Trimmed , broken scrap plastic', 'plastic chips: Trimmed , broken scrap plastic', 'EN', 'Material', '15 01 02', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('4', 'Turkey', 'A', '2', 'Pressed scrap metal', 'Pressed scrap metal', 'EN', 'Material', '15 01 04', 'Have', '5', 'tons', 'day', 'Ak Geri Dönü?üm: Can take non-hazardous metal waste;\nÖzvar Endüstriyel At?k: Can give to MKE;\nK?vanç Makine: They need steel scrap material as raw material in melting process;');
INSERT INTO `workshop_items2_full` VALUES ('5', 'Turkey', 'A', '2', 'Pressed scrap paper: Waste recycling facilities for paper mills or considered as intermediate products or raw materials', 'Pressed scrap paper: Waste recycling facilities for paper mills or considered as intermediate products or raw materials', 'EN', 'Material', '12 01 03', 'Have', '30', 'tons', 'day', 'ESOGÜ: Wastes as copper can be used for ceramic surface polishing (Doç. Dr. Çelik);');
INSERT INTO `workshop_items2_full` VALUES ('6', 'Turkey', 'A', '2', 'Industrial packaging, metal / plastic barrels , drums IBC tank: To prepare for re-use', 'Industrial packaging, metal / plastic barrels , drums IBC tank: To prepare for re-use', 'EN', 'Tools', '15 01 05', 'Have', null, null, null, 'Collecting packaging waste in the zone. 1000 tons/month');
INSERT INTO `workshop_items2_full` VALUES ('7', 'Turkey', 'A', '2', 'Pallets , big bags , sacks: Big bags and bags for use in food products.', 'Pallets , big bags , sacks: Big bags and bags for use in food products.', 'EN', 'Tools', '15 01 05', 'Have', null, null, null, 'Waste from the heat treatment furnaces. 1 ton');
INSERT INTO `workshop_items2_full` VALUES ('8', 'Turkey', 'A', '2', 'Chip powder', 'Chip powder', 'EN', 'Material', '15 01 03', 'Have', null, null, null, 'To be used in energy production and disposal facilities. 60.000 tons/year.');
INSERT INTO `workshop_items2_full` VALUES ('9', 'Turkey', 'A', '2', 'Pallet,  syrup tank: Plastic pallets suitable for food and syrup tank', 'Pallet,  syrup tank: Plastic pallets suitable for food and syrup tank', 'EN', 'Tools', '15 01 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('10', 'Turkey', 'A', '2', 'Home thrash', 'Home thrash', 'EN', 'Material', '20 03 01', 'Have', '250', 'tons', 'month', null);
INSERT INTO `workshop_items2_full` VALUES ('11', 'Turkey', 'A', '2', 'Packing certification services', 'Packing certification services', 'EN', 'Service', '02 01 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('12', 'Turkey', 'A', '2', 'consultancy: Companies to be given advice on waste management and system', 'consultancy: Companies to be given advice on waste management and system', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Requiring consultancy about market analysis, marketing and sales strategies');
INSERT INTO `workshop_items2_full` VALUES ('13', 'Turkey', 'A', '2', 'Packaging waste: Paper , plastic, pallets , glass, metal , composites can be used as raw material in wood packaging waste facility .', 'Packaging waste: Paper , plastic, pallets , glass, metal , composites can be used as raw material in wood packaging waste facility .', 'EN', 'Material', '15 01 05', 'Want', '10000', 'tons', 'year', null);
INSERT INTO `workshop_items2_full` VALUES ('14', 'Turkey', 'A', '2', 'Waste paper', 'Waste paper', 'EN', 'Material', '15 01 01', 'Want', '40', 'tons', 'day', 'Sawdust and thin chip waste from workshops and wood processing factories. 20.000 tons/year');
INSERT INTO `workshop_items2_full` VALUES ('15', 'Turkey', 'A', '2', 'Waste pallet', 'Waste pallet', 'EN', 'Tools', '15 01 06', 'Want', '5', 'tons', 'day', null);
INSERT INTO `workshop_items2_full` VALUES ('16', 'Turkey', 'A', '2', 'Paper, cardboard, wood', 'Paper, cardboard, wood', 'EN', 'Material', '15 01 01', 'Want', '1', 'tons', 'year', 'Any environmental measurements and analyzes are made (Accredited-?zmit)');
INSERT INTO `workshop_items2_full` VALUES ('17', 'Turkey', 'A', '2', 'Deformed plastic bags', 'Deformed plastic bags', 'EN', 'Material', '15 01 02', 'Want', '10', 'Kg', 'month', 'Metal-plastic barrels, drums, IBC tank');
INSERT INTO `workshop_items2_full` VALUES ('18', 'Turkey', 'A', '2', 'Crushed styrofoam: Used as raw materials taking into big bags.', 'Crushed styrofoam: Used as raw materials taking into big bags.', 'EN', 'Material', '12 01 05', 'Want', null, null, null, 'Providing services on the transport of hazardous waste');
INSERT INTO `workshop_items2_full` VALUES ('19', 'Turkey', 'A', '2', 'Non-hazardous metal waste , wood chips', 'Non-hazardous metal waste , wood chips', 'EN', 'Material', '17 02 01', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('20', 'Turkey', 'A', '2', 'scrap plastic', 'scrap plastic', 'EN', 'Material', '15 01 02', 'Want', '10', 'tons', 'day', null);
INSERT INTO `workshop_items2_full` VALUES ('21', 'Turkey', 'A', '2', 'Machinery / Equipment', 'Machinery / Equipment', 'EN', 'Tools', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('22', 'Turkey', 'A', '2', 'Support programs related consultancy: They want to get advice about the financial support.', 'Support programs related consultancy: They want to get advice about the financial support.', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('23', 'Turkey', 'A', '2', 'consultancy: EU funds , international funds to develop the possibilities of reaching common and processes. Industrial Symbiosis scope to connect with overseas companies.', 'consultancy: EU funds , international funds to develop the possibilities of reaching common and processes. Industrial Symbiosis scope to connect with overseas companies.', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('24', 'Turkey', 'A', '2', 'Training of welders for welding manufacturing', 'Training of welders for welding manufacturing', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Metal waste of the manufacturing industry and consultancy on its use in the ceramic industry.\n\nConsultancy on the production of methane from animal waste');
INSERT INTO `workshop_items2_full` VALUES ('25', 'Turkey', 'A', '3', 'waste paper: Creates the risk of fire scattered (without bales) with baling  delivered to the paper mill scrap of paper', 'waste paper: Creates the risk of fire scattered (without bales) with baling  delivered to the paper mill scrap of paper', 'EN', 'Material', '15 01 01', 'Have', '16', 'tons', 'month', null);
INSERT INTO `workshop_items2_full` VALUES ('26', 'Turkey', 'A', '3', 'Wooden pallets waste: Wood pallets are re-evaluated', 'Wooden pallets waste: Wood pallets are re-evaluated', 'EN', 'Tools', '15 01 03', 'Have', '2', 'tons', 'month', null);
INSERT INTO `workshop_items2_full` VALUES ('27', 'Turkey', 'A', '3', 'Water-based paint waste and offset: Waste from water-based paints are coming out of the printing plant', 'Water-based paint waste and offset: Waste from water-based paints are coming out of the printing plant', 'EN', 'Material', '08 01 12', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('28', 'Turkey', 'A', '3', 'Unused corrugated boxes and sheets: Used cardboard boxes in use are evaluated for use in the apparel business (must be plain )', 'Unused corrugated boxes and sheets: Used cardboard boxes in use are evaluated for use in the apparel business (must be plain )', 'EN', 'Material', '15 01 01', 'Want', '2', 'tons', 'month', null);
INSERT INTO `workshop_items2_full` VALUES ('29', 'Turkey', 'A', '3', 'Heat-resistant waste oil: Hot oil boilers with natural gas which is used for heat transfer', 'Heat-resistant waste oil: Hot oil boilers with natural gas which is used for heat transfer', 'EN', 'Material', '12 01 10', 'Want', '2', 'tons', null, 'Alpsan Makine: Reuse of the waste paper and paper packaging waste in production of biofuels');
INSERT INTO `workshop_items2_full` VALUES ('30', 'Turkey', 'A', '3', 'Nursery / Kindergarten: In OSB staff in a kindergarten or nursery children can go . Hours to prevent losses will facilitate the accessibility to children.', 'Nursery / Kindergarten: In OSB staff in a kindergarten or nursery children can go . Hours to prevent losses will facilitate the accessibility to children.', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Recycling of qualified and hazardous wastes and their introduction to the market (tins, cans, barrels, and oil contaminated materials, metal). 4000 tons/year');
INSERT INTO `workshop_items2_full` VALUES ('31', 'Turkey', 'A', '3', 'Education: Corporate Governance, HR and training support for R&D', 'Education: Corporate Governance, HR and training support for R&D', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Recycling of any kind of electronic waste. 2000 tons/year');
INSERT INTO `workshop_items2_full` VALUES ('32', 'Turkey', 'A', '3', 'Packing certification services', 'Packing certification services', 'EN', 'Service', '12 01 03', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('33', 'Turkey', 'A', '4', 'Iron filings', 'Iron filings', 'EN', 'Material', '16 01 17', 'Have', '10', 'tons', 'year', null);
INSERT INTO `workshop_items2_full` VALUES ('34', 'Turkey', 'A', '4', 'Iron and steel scrap', 'Iron and steel scrap', 'EN', 'Material', '16 01 17', 'Have', '40', 'tons', 'year', 'An alternative fuel for use in cement production. Can take 1500-2000 tons / year capacity.');
INSERT INTO `workshop_items2_full` VALUES ('35', 'Turkey', 'A', '4', 'Packaging waste and used wood pellets', 'Packaging waste and used wood pellets', 'EN', 'Tools', '15 01 03', 'Have', '1', 'tons', 'year', 'ICT, aircraft maintenance, general education and vocational courses for teachers');
INSERT INTO `workshop_items2_full` VALUES ('36', 'Turkey', 'A', '4', 'Iron slag', 'Iron slag', 'EN', 'Material', '16 01 17', 'Have', '1', 'tons', 'year', 'Lider Teknoloji can provide support on software');
INSERT INTO `workshop_items2_full` VALUES ('37', 'Turkey', 'A', '4', 'used cans', 'used cans', 'EN', 'Tools', '16 01 18', 'Have', '200', 'pcs', 'year', null);
INSERT INTO `workshop_items2_full` VALUES ('38', 'Turkey', 'A', '4', 'Packing certification services', 'Packing certification services', 'EN', 'Service', '16 01 18', 'Want', null, null, null, 'Industrial hazardous waste. 1 ton / month (variable).');
INSERT INTO `workshop_items2_full` VALUES ('39', 'Turkey', 'A', '5', 'Machinery / Equipment: We supply the means of production to companies engaged in the production of biomass .', 'Machinery / Equipment: We supply the means of production to companies engaged in the production of biomass .', 'EN', 'Service', '16 02 16', 'Have', null, null, null, 'It is the sedimentary section of the refectory wastewater deposited on top after passing the grease trap. Can be used in soap industry. 2,5 tons/month');
INSERT INTO `workshop_items2_full` VALUES ('40', 'Turkey', 'A', '5', 'Paper waste: For use as biofuel . Eco-friendly form of fuel (pellets ) will be brought .', 'Paper waste: For use as biofuel . Eco-friendly form of fuel (pellets ) will be brought .', 'EN', 'Energy', '15 01 01', 'Want', null, null, null, 'Sabiha Gökçen MTAL: Providing vocational training services to public and private sectors.\nTÜLOMSA?: Accredited educational services for welder training (17024)');
INSERT INTO `workshop_items2_full` VALUES ('41', 'Turkey', 'A', '5', 'Wood packaging waste', 'Wood packaging waste', 'EN', 'Material', '15 01 03', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('42', 'Turkey', 'A', '5', 'Paper, cardboard, wood', 'Paper, cardboard, wood', 'EN', 'Material', '15 01 01', 'Want', null, null, null, 'Joint projects can be done in waste and hazardous waste management, management of AEEE, sludge decision support systems, life cycle analysis');
INSERT INTO `workshop_items2_full` VALUES ('43', 'Turkey', 'A', '5', 'Sawmill dust and shavings', 'Sawmill dust and shavings', 'EN', 'Material', '15 01 01', 'Want', null, null, null, 'Technical consulting servi?ces in erosion control, landscaping issues and reducing carbon footprint. Can be useful for companies in organized industrial zone.');
INSERT INTO `workshop_items2_full` VALUES ('44', 'Turkey', 'A', '5', 'Domestic / food waste: To see their food waste .', 'Domestic / food waste: To see their food waste .', 'EN', 'Material', '02 03 04', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('45', 'Turkey', 'A', '5', 'Packaging waste', 'Packaging waste', 'EN', 'Material', '15 01 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('46', 'Turkey', 'A', '5', 'Forest products manufacturing waste, wood waste: For use as biofuel .', 'Forest products manufacturing waste, wood waste: For use as biofuel .', 'EN', 'Energy', '03 01 01', 'Want', null, null, null, 'Big bags and bags for use in food products');
INSERT INTO `workshop_items2_full` VALUES ('47', 'Turkey', 'A', '6', 'Lab: Solid waste with the relevant laboratories in existing GC-MS elemental analysis, calorimetry , oxygen permeability , extruders, including laboratory analysis on the FCP- MS instrument can be analyzed to support research projects . ( A part of the Lab is accredited )', 'Lab: Solid waste with the relevant laboratories in existing GC-MS elemental analysis, calorimetry , oxygen permeability , extruders, including laboratory analysis on the FCP- MS instrument can be analyzed to support research projects . ( A part of the Lab is accredited )', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Recycling of qualified and hazardous wastes and their introduction to the market (tins, cans, barrels, and oil contaminated materials, metal). 4000 tons/year');
INSERT INTO `workshop_items2_full` VALUES ('48', 'Turkey', 'A', '6', 'Waste and hazardous waste management: Waste and hazardous waste management, management of AEEE , sludge decision support systems , life cycle analysis, etc. It can be carried out joint projects with companies on issues .', 'Waste and hazardous waste management: Waste and hazardous waste management, management of AEEE , sludge decision support systems , life cycle analysis, etc. It can be carried out joint projects with companies on issues .', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Iron, chromium, manganese content lower materials, Sand, quartzite, limestone, marble, iron, aluminum, sodium and potassium content high materials');
INSERT INTO `workshop_items2_full` VALUES ('49', 'Turkey', 'A', '6', 'Expertise: Industrial Symbiosis and advice on waste management , project management and so on. the delivery of services', 'Expertise: Industrial Symbiosis and advice on waste management , project management and so on. the delivery of services', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'For use in road construction');
INSERT INTO `workshop_items2_full` VALUES ('50', 'Turkey', 'A', '6', 'Rind: Propose: Biopolymer manufacturing available from orange peel , you can contact the packing company', 'Rind: Propose: Biopolymer manufacturing available from orange peel , you can contact the packing company', 'EN', 'Material', '02 03 04', 'Want', '1', 'tons', 'week', 'An alternative fuel for use in cement production. 1500-2000 tons/year');
INSERT INTO `workshop_items2_full` VALUES ('51', 'Turkey', 'A', '7', 'Terracotta tile waste: Baked and glazed tiles can not be milled is angopl high strength . Terracotta tile waste, specific dimensions and surfaces are smooth .', 'Terracotta tile waste: Baked and glazed tiles can not be milled is angopl high strength . Terracotta tile waste, specific dimensions and surfaces are smooth .', 'EN', 'Material', '17 01 06', 'Have', null, null, null, 'Technical consulting servi?ces in erosion control, landscaping issues and reducing carbon footprint. Can be useful for companies in organized industrial zone.');
INSERT INTO `workshop_items2_full` VALUES ('52', 'Turkey', 'A', '7', 'Ceramic raw materials and production waste', 'Ceramic raw materials and production waste', 'EN', 'Material', '17 01 06', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('53', 'Turkey', 'A', '7', 'Natural mineral waste', 'Natural mineral waste', 'EN', 'Material', '06 13 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('54', 'Turkey', 'A', '7', 'Waste powder coating', 'Waste powder coating', 'EN', 'Material', '12 01 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('55', 'Turkey', 'A', '7', 'Sewage sludge: Analysis needs to be done .', 'Sewage sludge: Analysis needs to be done .', 'EN', 'Material', '17 01 07', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('56', 'Turkey', 'A', '7', 'Water-based paint waste and offset: If it can be in a dry form .', 'Water-based paint waste and offset: If it can be in a dry form .', 'EN', 'Material', '08 01 12', 'Want', null, null, null, 'Fruit peel, fruit syrups. 1 ton/week');
INSERT INTO `workshop_items2_full` VALUES ('57', 'Turkey', 'A', '7', 'Waste and hazardous waste management', 'Waste and hazardous waste management', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('58', 'Turkey', 'A', '8', 'Milek sludge: The compressor is due to the production. It stands out from most other businesses in Eskisehir business . Bakelite is dense mud. Iron is also found inside .', 'Milek sludge: The compressor is due to the production. It stands out from most other businesses in Eskisehir business . Bakelite is dense mud. Iron is also found inside .', 'EN', 'Material', '12 01 99', 'Have', '110', 'tons', 'year', null);
INSERT INTO `workshop_items2_full` VALUES ('59', 'Turkey', 'A', '8', 'Waste metal: Sheet metal wastes ( iron, copper , aluminum ) .Going to the landfill.', 'Waste metal: Sheet metal wastes ( iron, copper , aluminum ) .Going to the landfill.', 'EN', 'Material', '12 01 03', 'Have', '6500', 'tons', 'year', null);
INSERT INTO `workshop_items2_full` VALUES ('60', 'Turkey', 'A', '8', 'Industrial packaging, metal / plastic barrels , drums IBC tank: Industrial packaging waste and others', 'Industrial packaging, metal / plastic barrels , drums IBC tank: Industrial packaging waste and others', 'EN', 'Tools', '12 01 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('61', 'Turkey', 'A', '8', 'Cold Rolled Steel, Steel Scrap', 'Cold Rolled Steel, Steel Scrap', 'EN', 'Material', '12 01 01', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('62', 'Turkey', 'A', '8', 'Aluminum sludge', 'Aluminum sludge', 'EN', 'Material', '12 01 04', 'Have', null, null, null, 'A new company is looking for financial support and technical consultancy on quality certification and machinery and equipment subjects');
INSERT INTO `workshop_items2_full` VALUES ('63', 'Turkey', 'A', '8', 'Waste powder coating: They are caused by the dye house waste .', 'Waste powder coating: They are caused by the dye house waste .', 'EN', 'Material', '12 01 99', 'Have', '45', 'tons', 'year', null);
INSERT INTO `workshop_items2_full` VALUES ('64', 'Turkey', 'A', '8', 'hydraulic oil: Anhydrous, 1st category . To Koza sold at the moment.', 'hydraulic oil: Anhydrous, 1st category . To Koza sold at the moment.', 'EN', 'Material', '12 01 10', 'Have', '55', 'tons', 'year', null);
INSERT INTO `workshop_items2_full` VALUES ('65', 'Turkey', 'A', '8', 'Industrial packaging, metal / plastic barrels , drums IBC tank: Waste recycling industrial packaging and other services', 'Industrial packaging, metal / plastic barrels , drums IBC tank: Waste recycling industrial packaging and other services', 'EN', 'Tools', '12 01 10', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('66', 'Turkey', 'A', '8', 'Alcaly: For use in the treatment plant', 'Alcaly: For use in the treatment plant', 'EN', 'Material', '20 01 15', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('67', 'Turkey', 'A', '8', 'Acid(HCl): For use in the treatment plant', 'Acid(HCl): For use in the treatment plant', 'EN', 'Material', '12 01 10', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('68', 'Turkey', 'A', '8', 'Expertise - laboratory services', 'Expertise - laboratory services', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('69', 'Turkey', 'A', '9', 'Rubber processing waste: 0-1mm arising from rubber recycling process emerging during the separation process of fine rubber powder and packaging waste,recycling of waste are not a resident .', 'Rubber processing waste: 0-1mm arising from rubber recycling process emerging during the separation process of fine rubber powder and packaging waste,recycling of waste are not a resident .', 'EN', 'Material', '19 12 04', 'Have', '1000', 'tons', 'year', null);
INSERT INTO `workshop_items2_full` VALUES ('70', 'Turkey', 'A', '9', 'Cellulose , cardboard, kraft cellulose such as cement bags', 'Cellulose , cardboard, kraft cellulose such as cement bags', 'EN', 'Material', '15 01 01', 'Have', null, null, null, '60.000 tons/year capacity for use in energy production and disposal plants');
INSERT INTO `workshop_items2_full` VALUES ('71', 'Turkey', 'A', '9', 'Pallets , big bags , sacks: Big bags and bags for use in food products.', 'Pallets , big bags , sacks: Big bags and bags for use in food products.', 'EN', 'Tools', '15 01 01', 'Have', null, null, null, 'Supports needed for setting-up the system of Municipality to collect the waste and starting the recycling process');
INSERT INTO `workshop_items2_full` VALUES ('72', 'Turkey', 'A', '9', 'waste paper: Exists as clippings', 'waste paper: Exists as clippings', 'EN', 'Material', '12 02 16', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('73', 'Turkey', 'A', '9', 'Integrated waste ( packaging - waste oil and Electronic Waste , etc.).', 'Integrated waste ( packaging - waste oil and Electronic Waste , etc.).', 'EN', 'Material', '12 02 16', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('74', 'Turkey', 'A', '9', 'mixed waste: The separation of industrial and domestic waste, used to convert energy from organic waste.', 'mixed waste: The separation of industrial and domestic waste, used to convert energy from organic waste.', 'EN', 'Material', '12 02 16', 'Want', '300', 'tons', 'day', null);
INSERT INTO `workshop_items2_full` VALUES ('75', 'Turkey', 'A', '9', 'Assessable mixed packaging waste: Considered the least amount of packaging waste collected from businesses within the boundaries of Odunpazari.', 'Assessable mixed packaging waste: Considered the least amount of packaging waste collected from businesses within the boundaries of Odunpazari.', 'EN', 'Material', '12 02 16', 'Want', '1000', 'tons', 'month', 'Already going to Istanbul. Analysis are available (dangerous), used in the elevator industry. Core sand composition: sio?. 4160 kg / year.');
INSERT INTO `workshop_items2_full` VALUES ('76', 'Turkey', 'A', '9', 'Cellulose , cardboard, kraft cellulose such as cement bags: Ekobord Yeni Nesil Levha of plate ( Bilecik) are the resources that it can get .', 'Cellulose , cardboard, kraft cellulose such as cement bags: Ekobord Yeni Nesil Levha of plate ( Bilecik) are the resources that it can get .', 'EN', 'Material', '12 02 16', 'Want', null, null, null, '200 units/year');
INSERT INTO `workshop_items2_full` VALUES ('77', 'Turkey', 'A', '9', 'Biomass machinery', 'Biomass machinery', 'EN', 'Service', '12 02 16', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('78', 'Turkey', 'A', '9', 'Service', 'Service', 'EN', 'Service', '12 02 16', 'Want', null, null, null, 'Joint projects can be done in waste and hazardous waste management, management of AEEE, sludge decision support systems, life cycle analysis');
INSERT INTO `workshop_items2_full` VALUES ('79', 'Turkey', 'A', '10', 'Paint sludge: Wet coating process resulting paint sludge .', 'Paint sludge: Wet coating process resulting paint sludge .', 'EN', 'Material', '99 99 99', 'Have', '9', 'tons', 'year', 'Supplying production tools for producers of biomass');
INSERT INTO `workshop_items2_full` VALUES ('80', 'Turkey', 'A', '11', 'Rack and cutting oil waste: By containing of waste oil disposal or use of the benches at work again', 'Rack and cutting oil waste: By containing of waste oil disposal or use of the benches at work again', 'EN', 'Material', '12 01 10', 'Have', '10', 'barrels', 'year', 'H?zlan A.?: Can take machinery oils, but licence is needed');
INSERT INTO `workshop_items2_full` VALUES ('81', 'Turkey', 'A', '11', 'Staff transport: To establish a common transport system for employees.', 'Staff transport: To establish a common transport system for employees.', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('82', 'Turkey', 'A', '12', 'Cellulose , cardboard, kraft cellulose such as cement bags', 'Cellulose , cardboard, kraft cellulose such as cement bags', 'EN', 'Material', '15 01 01', 'Have', null, null, null, 'An alternative fuel for use in cement production. 60.000 tons/year capacity');
INSERT INTO `workshop_items2_full` VALUES ('83', 'Turkey', 'A', '12', 'Refractory materials', 'Refractory materials', 'EN', 'Material', '10 13 14', 'Have', null, null, null, 'Going to Benli Dönü?üm. The company is sorting the wastes and logistics');
INSERT INTO `workshop_items2_full` VALUES ('84', 'Turkey', 'A', '12', 'waste Heat: It emerges from clinker production line. Ex temperature is 200-250 degrees.', 'waste Heat: It emerges from clinker production line. Ex temperature is 200-250 degrees.', 'EN', 'Energy', '99 99 99', 'Have', '300000', 'm3', 'hour', '');
INSERT INTO `workshop_items2_full` VALUES ('85', 'Turkey', 'A', '12', 'Lab: Raw materials for cement production , contribution and analysis of all kinds of materials can be used as fuel.', 'Lab: Raw materials for cement production , contribution and analysis of all kinds of materials can be used as fuel.', 'EN', 'Service', '10 13 99', 'Have', null, null, null, 'Any kind of recycling of electronic waste is done. 2000 tons/year');
INSERT INTO `workshop_items2_full` VALUES ('86', 'Turkey', 'A', '12', 'Gypsum waste: To use alternative raw materials in cement production', 'Gypsum waste: To use alternative raw materials in cement production', 'EN', 'Material', '17 08 02', 'Want', '125000', 'tons', 'year', null);
INSERT INTO `workshop_items2_full` VALUES ('87', 'Turkey', 'A', '12', 'Clay -containing waste: To use alternative raw materials in cement production', 'Clay -containing waste: To use alternative raw materials in cement production', 'EN', 'Material', '01 04 09', 'Want', '250000', 'tons', 'year', null);
INSERT INTO `workshop_items2_full` VALUES ('88', 'Turkey', 'A', '12', 'Limestone containing waste: To use alternative raw materials in cement production', 'Limestone containing waste: To use alternative raw materials in cement production', 'EN', 'Material', '10 13 01', 'Want', '250000', 'tons', 'year', 'A new company is looking for financial supports and technical consultancy on quality certification and machinery and equipment subjects');
INSERT INTO `workshop_items2_full` VALUES ('89', 'Turkey', 'A', '12', 'Sewage sludge: Alternative fuel for use in cement production. Maximum 10% humidity and a minimum 2500 kcal / kg should be in value.', 'Sewage sludge: Alternative fuel for use in cement production. Maximum 10% humidity and a minimum 2500 kcal / kg should be in value.', 'EN', 'Material', '17 01 07', 'Want', null, null, null, 'Slag and foundry sand of K?vanç Makine and Deniz Döküm can be used.\nAnalysis is needed for the waste of Anka');
INSERT INTO `workshop_items2_full` VALUES ('90', 'Turkey', 'A', '12', 'Construction and demolition waste', 'Construction and demolition waste', 'EN', 'Material', '17 09 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('91', 'Turkey', 'A', '12', 'Terracotta tile waste', 'Terracotta tile waste', 'EN', 'Material', '17 01 06', 'Want', null, null, null, 'Ak Geri Dönü?üm: Can be used as raw material in paper, plastic, pallets, glass, metal, composites and wood packaging waste facilities. 10.000 tons/year\nAk Oluklu Ambalaj: Wood pallets can be re-used.\nTÜLOMSA?: Wood pallet wastes');
INSERT INTO `workshop_items2_full` VALUES ('92', 'Turkey', 'A', '12', ' Gypsum Mold Waste', ' Gypsum Mold Waste', 'EN', 'Material', '17 08 02', 'Want', null, null, null, 'Slag and foundry sand of K?vanç Makine and Deniz Döküm can be used.\nAnalysis is needed.');
INSERT INTO `workshop_items2_full` VALUES ('93', 'Turkey', 'A', '12', 'Wooden pallets, waste', 'Wooden pallets, waste', 'EN', 'Tools', '15 01 03', 'Want', null, null, null, '?lksem Mühendislik: Proje yazma, kalite yönetimi ve yönetici e?itimleri\nSabiha Gökçen MTAL: Mesleki e?itim');
INSERT INTO `workshop_items2_full` VALUES ('94', 'Turkey', 'A', '12', 'wood pellets', 'wood pellets', 'EN', 'Tools', '12 01 10', 'Want', null, null, null, 'AB fonlar? uluslaras? ortaklara ve süreci geli?tirebilecek fonlara ula?ma imkan?. Endüstiryel Simbiyoz kapsam?nda yurt d??? firmalar ile ba?lant? kurma.');
INSERT INTO `workshop_items2_full` VALUES ('95', 'Turkey', 'A', '12', 'Ceramic fractures', 'Ceramic fractures', 'EN', 'Material', '10 02 02', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('96', 'Turkey', 'A', '12', 'Food and textile waste', 'Food and textile waste', 'EN', 'Material', '02 02 03', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('97', 'Turkey', 'A', '12', 'Core and foundry sand', 'Core and foundry sand', 'EN', 'Material', '10 09 08', 'Want', null, null, null, 'Orman Bölge Müdürlü?ü: Consultancy services on carbon footprint reducing, erosion control and landscaping issues. Can be useful for companies located in organized industrial zone.\nSabiha Gökçen MTAL: Project partnership');
INSERT INTO `workshop_items2_full` VALUES ('98', 'Turkey', 'A', '12', 'Slag waste', 'Slag waste', 'EN', 'Material', '10 02 02', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('99', 'Turkey', 'A', '12', 'Filter press sludge', 'Filter press sludge', 'EN', 'Material', '17 01 07', 'Want', null, null, null, 'Support for systematically collecting the packing wastes from houses, setting-up a collection methodology and bringing it into the recycling process');
INSERT INTO `workshop_items2_full` VALUES ('100', 'Turkey', 'A', '12', 'Household waste , solid waste, waste oils from construction machinery', 'Household waste , solid waste, waste oils from construction machinery', 'EN', 'Material', '20 01 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('101', 'Turkey', 'A', '12', 'Debris waste', 'Debris waste', 'EN', 'Material', '17 09 03', 'Want', null, null, null, '?lksem Mühendislik: On behalf of its consultant company');
INSERT INTO `workshop_items2_full` VALUES ('102', 'Turkey', 'A', '12', 'Natural mineral waste: Iron, chromium, manganese content lower materials\nSand , quartzite, limestone , marble, iron, aluminum, sodium and potassium content is high materials.', 'Natural mineral waste: Iron, chromium, manganese content lower materials\nSand , quartzite, limestone , marble, iron, aluminum, sodium and potassium content is high materials.', 'EN', 'Material', '06 13 99', 'Want', null, null, null, 'Vocational trainings on ICT and aircraft maintenance for teachers');
INSERT INTO `workshop_items2_full` VALUES ('103', 'Turkey', 'A', '12', 'Solids: Alternative fuel for use in cement production', 'Solids: Alternative fuel for use in cement production', 'EN', 'Material', '99 99 99', 'Want', '60000', 'tons', 'year', null);
INSERT INTO `workshop_items2_full` VALUES ('104', 'Turkey', 'A', '12', 'Solvent waste: Alternative fuel for use in cement production', 'Solvent waste: Alternative fuel for use in cement production', 'EN', 'Material', '99 99 99', 'Want', '1750', 'tons', 'year', 'Alpsan Makine: To use as biofuels');
INSERT INTO `workshop_items2_full` VALUES ('105', 'Turkey', 'A', '12', 'waste oil: Alternative fuel for use in cement production', 'waste oil: Alternative fuel for use in cement production', 'EN', 'Material', '12 01 10', 'Want', '1750', 'tons', 'year', 'Ekobord Yeni Nesil Levha Co. (Bilecik) can take this. 60 tons/month');
INSERT INTO `workshop_items2_full` VALUES ('106', 'Turkey', 'A', '12', 'hydraulic oil', 'hydraulic oil', 'EN', 'Material', '12 01 10', 'Want', null, null, null, 'fSawdust and wood chip waste from workshops and factories working on wood processing. 20.000 tons/year');
INSERT INTO `workshop_items2_full` VALUES ('107', 'Turkey', 'A', '12', 'Testing / Analysis: Chemical and physical analysis. Mineralogical analysis ( XRD and XRF ) service. Accredited ceramic final product testing. Boron analysis services provided.', 'Testing / Analysis: Chemical and physical analysis. Mineralogical analysis ( XRD and XRF ) service. Accredited ceramic final product testing. Boron analysis services provided.', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('108', 'Turkey', 'A', '12', 'Project and Consultancy Services: Across the public and private sector organizations in the preparation of national and international projects and cooperation in the implementation', 'Project and Consultancy Services: Across the public and private sector organizations in the preparation of national and international projects and cooperation in the implementation', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Endel A.?: Can take aggregate, should be analyzed\nH?zlan: Company must be licensed for core sand exchange');
INSERT INTO `workshop_items2_full` VALUES ('109', 'Turkey', 'A', '12', 'Lab', 'Lab', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Foundry slags from hot rolling mills can be used as feeder. 1 ton/month');
INSERT INTO `workshop_items2_full` VALUES ('110', 'Turkey', 'A', '12', 'Expertise', 'Expertise', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Regularly');
INSERT INTO `workshop_items2_full` VALUES ('111', 'Turkey', 'A', '12', 'Testing / Analysis', 'Testing / Analysis', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('112', 'Turkey', 'A', '12', 'Project and Consultancy Services', 'Project and Consultancy Services', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'To use for road construction of science works office');
INSERT INTO `workshop_items2_full` VALUES ('113', 'Turkey', 'A', '12', 'Saplings and Seeds', 'Saplings and Seeds', 'EN', 'Material', '02 01 03', 'Want', null, null, null, 'Can be used in Municipalities and rural services directorate ');
INSERT INTO `workshop_items2_full` VALUES ('114', 'Turkey', 'A', '12', 'consultancy', 'consultancy', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Can be taken as one of the main components for production of concrete tiles. 15.000 tons');
INSERT INTO `workshop_items2_full` VALUES ('115', 'Turkey', 'A', '12', 'Education and Human Resource Services', 'Education and Human Resource Services', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Can be taken as one of the main components for production of concrete tiles. 20.000 tons');
INSERT INTO `workshop_items2_full` VALUES ('116', 'Turkey', 'A', '12', 'Training of welders for welding manufacturing', 'Training of welders for welding manufacturing', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('117', 'Turkey', 'A', '12', 'Packing certification services', 'Packing certification services', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Ak Geri Dönü?üm: Pressed scrap metal\nArçelik: 6500 tons/year');
INSERT INTO `workshop_items2_full` VALUES ('118', 'Turkey', 'A', '13', 'Crushed styrofoam', 'Crushed styrofoam', 'EN', 'Material', '12 01 05', 'Have', '1', 'm3', 'per month', null);
INSERT INTO `workshop_items2_full` VALUES ('119', 'Turkey', 'A', '13', 'Facing sand', 'Facing sand', 'EN', 'Material', '12 01 99', 'Have', '20', 'tons', 'per month', 'Metal scrap can be used in the foundry industry');
INSERT INTO `workshop_items2_full` VALUES ('120', 'Turkey', 'A', '13', 'Seedlings soil', 'Seedlings soil', 'EN', 'Material', '12 01 99', 'Have', null, null, null, 'Wastes from the manufacturing process');
INSERT INTO `workshop_items2_full` VALUES ('121', 'Turkey', 'A', '13', 'Asphalt and other road-building materials', 'Asphalt and other road-building materials', 'EN', 'Material', '12 01 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('122', 'Turkey', 'A', '13', 'Assessable mixed packaging waste', 'Assessable mixed packaging waste', 'EN', 'Material', '15 01 06', 'Have', null, null, null, 'Alpsan Makine: Can take food waste');
INSERT INTO `workshop_items2_full` VALUES ('123', 'Turkey', 'A', '13', 'Cutting fluids: Hazardous waste is set . It consists of tramp oil and coolant ', 'Cutting fluids: Hazardous waste is set . It consists of tramp oil and coolant ', 'EN', 'Material', '12 01 10', 'Have', '1', 'BBL', 'per month', null);
INSERT INTO `workshop_items2_full` VALUES ('124', 'Turkey', 'A', '13', 'descaling: There may be waste from the heat treatment furnaces.', 'descaling: There may be waste from the heat treatment furnaces.', 'EN', 'Energy', '99 99 99', 'Want', '1', 'tons', 'per month', null);
INSERT INTO `workshop_items2_full` VALUES ('125', 'Turkey', 'A', '13', 'Aluminum dross: While molding material used in the production of feeder liners in foundry.', 'Aluminum dross: While molding material used in the production of feeder liners in foundry.', 'EN', 'Material', '17 04 02', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('126', 'Turkey', 'A', '13', 'Sprue: Casting the production of smelter slag can be used in the feeder sleeve.', 'Sprue: Casting the production of smelter slag can be used in the feeder sleeve.', 'EN', 'Material', '12 01 03', 'Want', '1', 'tons', 'per month', 'Orman Bölge Md: Can take trainings\nAk Oluklu Ambalaj: Institutionalization, HR and R&D supports and trainings\nTÜLOMSA?: Production and management processes trainings');
INSERT INTO `workshop_items2_full` VALUES ('127', 'Turkey', 'A', '13', 'Copper slag', 'Copper slag', 'EN', 'Material', '17 04 01', 'Want', null, null, null, 'ES Anka Havac?l?k: A new company is looking for financial supports and technical consultancy on quality certification and machinery and equipment subjects\nAk Geri Dönü?üm: Needs support on financial incentives\nP?nar Süt: Needs consultancy on wastewater treatment plant management and related investments');
INSERT INTO `workshop_items2_full` VALUES ('128', 'Turkey', 'A', '13', 'Refractory materials: Refractory materials coming from the quarries used ( stone wool, lining etc . ).', 'Refractory materials: Refractory materials coming from the quarries used ( stone wool, lining etc . ).', 'EN', 'Material', '16 11 06', 'Want', '1', 'tons', 'per month', 'Project partnership');
INSERT INTO `workshop_items2_full` VALUES ('129', 'Turkey', 'A', '13', 'waste paper: For use in the manufacture of feeder liners .', 'waste paper: For use in the manufacture of feeder liners .', 'EN', 'Material', '15 01 01', 'Want', '1', 'm3', 'per month', null);
INSERT INTO `workshop_items2_full` VALUES ('130', 'Turkey', 'A', '13', 'White cement dust: For use in making foundry molds .', 'White cement dust: For use in making foundry molds .', 'EN', 'Material', '10 13 06', 'Want', '1', 'tons', 'per month', 'Resin-impregnated paper. Available in various colors and sizes 0.5 / 0.7 micron pieces');
INSERT INTO `workshop_items2_full` VALUES ('131', 'Turkey', 'A', '13', 'billethead of scrap metal: Arising from the mill.', 'billethead of scrap metal: Arising from the mill.', 'EN', 'Material', '17 04 07', 'Want', null, null, null, 'Waste from the ceramics factory. Inert, 17% humidity. 1000 tons/month');
INSERT INTO `workshop_items2_full` VALUES ('132', 'Turkey', 'A', '13', 'wood Pallet', 'wood Pallet', 'EN', 'Tools', '12 01 10', 'Want', null, null, null, 'Sawdust and wood chip waste from workshops and factories working on wood processing. 20.000 tons/year');
INSERT INTO `workshop_items2_full` VALUES ('133', 'Turkey', 'A', '13', 'Destructive / Nondestructive Laboratory Services', 'Destructive / Nondestructive Laboratory Services', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('134', 'Turkey', 'A', '13', 'Education', 'Education', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Consultancy services on machinery, system, software, hardware and process development');
INSERT INTO `workshop_items2_full` VALUES ('135', 'Turkey', 'A', '14', 'PE-PP granules and plastic: Semi -finished plastic manufacturing firms as granules or industrial trash bags can be supplied .', 'PE-PP granules and plastic: Semi -finished plastic manufacturing firms as granules or industrial trash bags can be supplied .', 'EN', 'Material', '17 02 03', 'Have', null, null, null, 'Chemical and physical analysis. Mineralogical analysis (XRD and XRF) service. Accredited ceramic final product tests. Boron analysis services');
INSERT INTO `workshop_items2_full` VALUES ('136', 'Turkey', 'A', '14', 'wood Pellet: Stowage and shipment of materials can be produced for use in industrial companies.', 'wood Pellet: Stowage and shipment of materials can be produced for use in industrial companies.', 'EN', 'Tools', '15 01 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('137', 'Turkey', 'A', '14', 'Wood , timber , fertilizer', 'Wood , timber , fertilizer', 'EN', 'Material', '03 01 01', 'Have', null, null, null, 'To establish a common transport system for employees');
INSERT INTO `workshop_items2_full` VALUES ('138', 'Turkey', 'A', '14', 'Pallets , big bags , sacks', 'Pallets , big bags , sacks', 'EN', 'Tools', '12 01 10', 'Have', null, null, null, 'Can work with Seranit and ESÇEV');
INSERT INTO `workshop_items2_full` VALUES ('139', 'Turkey', 'A', '14', 'Chip powder', 'Chip powder', 'EN', 'Material', '15 01 02', 'Have', null, null, null, 'Deformed plastic bags. 10 kg/month');
INSERT INTO `workshop_items2_full` VALUES ('140', 'Turkey', 'A', '14', 'Forest products manufacturing waste, wood waste', 'Forest products manufacturing waste, wood waste', 'EN', 'Material', '03 01 01', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('141', 'Turkey', 'A', '14', 'Waste PP- PE-PP: Ensuring the recovery of waste polyurethane and polypropylene plastic material used to obtain granular raw material and industrial bags is obtained.', 'Waste PP- PE-PP: Ensuring the recovery of waste polyurethane and polypropylene plastic material used to obtain granular raw material and industrial bags is obtained.', 'EN', 'Material', '17 02 03', 'Want', null, null, null, 'Licence needed');
INSERT INTO `workshop_items2_full` VALUES ('142', 'Turkey', 'A', '14', 'Wood waste: Recycled wood waste collected and recycled into the economy ', 'Wood waste: Recycled wood waste collected and recycled into the economy ', 'EN', 'Material', '03 01 01', 'Want', '1000', 'tons', 'per month', null);
INSERT INTO `workshop_items2_full` VALUES ('143', 'Turkey', 'A', '14', 'waste plastic', 'waste plastic', 'EN', 'Material', '17 02 03', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('144', 'Turkey', 'A', '14', 'Metal (brass), Aluminium - copper etc. Plastic derivatives', 'Metal (brass), Aluminium - copper etc. Plastic derivatives', 'EN', 'Material', '17 04 01', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('145', 'Turkey', 'A', '14', 'Paper, cardboard, wood', 'Paper, cardboard, wood', 'EN', 'Material', '20 01 01', 'Want', '900', 'kg', 'per year', 'For use as biofuel. At the planning stage');
INSERT INTO `workshop_items2_full` VALUES ('146', 'Turkey', 'A', '14', 'plastic chips', 'plastic chips', 'EN', 'Material', '12 01 05', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('147', 'Turkey', 'A', '14', 'Waste transport services', 'Waste transport services', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Tepeba?? Bld.: Can be discussed for road construction');
INSERT INTO `workshop_items2_full` VALUES ('148', 'Turkey', 'A', '15', 'Hazardous waste: According to the office arising from recycling packaging, glass , bottles, paper waste', 'Hazardous waste: According to the office arising from recycling packaging, glass , bottles, paper waste', 'EN', 'Material', '15 01 06', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('149', 'Turkey', 'A', '15', 'Electronic waste', 'Electronic waste', 'EN', 'Material', '16 02 16', 'Have', null, null, null, 'Can take 1 ton/month. Refractory materials coming from the quarries can be used (rockwool, lining etc.)');
INSERT INTO `workshop_items2_full` VALUES ('150', 'Turkey', 'A', '15', 'Pallets , big bags , sacks', 'Pallets , big bags , sacks', 'EN', 'Tools', '15 01 06', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('151', 'Turkey', 'A', '15', 'Packaging waste', 'Packaging waste', 'EN', 'Material', '15 01 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('152', 'Turkey', 'A', '15', 'Industrial packaging, metal / plastic barrels , IBC tank waste bins', 'Industrial packaging, metal / plastic barrels , IBC tank waste bins', 'EN', 'Tools', '19 12 12', 'Have', null, null, null, 'Tepeba?? Bld. can help in this issue');
INSERT INTO `workshop_items2_full` VALUES ('153', 'Turkey', 'A', '15', 'Hazardous / non-hazardous waste: For use in energy production and disposal facilities .', 'Hazardous / non-hazardous waste: For use in energy production and disposal facilities .', 'EN', 'Energy', '99 99 99', 'Want', '60000', 'tons', 'per year', 'Joint projects can be done in waste and hazardous waste management, management of AEEE, sludge decision support systems, life cycle analysis');
INSERT INTO `workshop_items2_full` VALUES ('154', 'Turkey', 'A', '15', 'Facing sand', 'Facing sand', 'EN', 'Material', '99 99 99', 'Want', null, null, null, 'Anka Toprak Ürünleri: Analysis should be done');
INSERT INTO `workshop_items2_full` VALUES ('155', 'Turkey', 'A', '15', 'Iron slag', 'Iron slag', 'EN', 'Material', '16 01 17', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('156', 'Turkey', 'A', '15', 'Iron and steel scrap', 'Iron and steel scrap', 'EN', 'Material', '16 01 17', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('157', 'Turkey', 'A', '15', 'Packaging waste and used wood pellets', 'Packaging waste and used wood pellets', 'EN', 'Material', '15 01 03', 'Want', null, null, null, 'ESO: Recycling is possible according to various characterization of casting and core sand');
INSERT INTO `workshop_items2_full` VALUES ('158', 'Turkey', 'A', '15', 'used cans', 'used cans', 'EN', 'Tools', '16 01 18', 'Want', null, null, null, 'Anka Toprak Ürünleri: Analysis needed');
INSERT INTO `workshop_items2_full` VALUES ('159', 'Turkey', 'A', '15', 'waste paper', 'waste paper', 'EN', 'Material', '15 01 01', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('160', 'Turkey', 'A', '15', 'Rind', 'Rind', 'EN', 'Material', '99 99 99', 'Want', null, null, null, 'ESO: Can be used as combustable after being pressed\nAlpsan Makine: Can use in future\nTÜLOMSA?: Can be used to absorve the waste oil');
INSERT INTO `workshop_items2_full` VALUES ('161', 'Turkey', 'A', '15', 'Electrical / electronic materials', 'Electrical / electronic materials', 'EN', 'Material', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('162', 'Turkey', 'A', '15', 'Filter press sludge', 'Filter press sludge', 'EN', 'Material', '17 01 07', 'Want', null, null, null, 'Can be used for Municipality parks and rural areas of Municipality');
INSERT INTO `workshop_items2_full` VALUES ('163', 'Turkey', 'A', '15', 'Household waste , solid waste, waste oils from construction machinery', 'Household waste , solid waste, waste oils from construction machinery', 'EN', 'Material', '12 01 10', 'Want', null, null, null, 'In planning stage. In order to use as bio-fuel');
INSERT INTO `workshop_items2_full` VALUES ('164', 'Turkey', 'A', '15', 'Contaminated waste oil\nBoric oil\nHydraulic oil', 'Contaminated waste oil\nBoric oil\nHydraulic oil', 'EN', 'Material', '99 99 99', 'Want', null, null, null, 'Odunpazar? Bld.: May workshop for footprint. Tepeba?? Bld.: Can be advised for rehabilitation of mines.');
INSERT INTO `workshop_items2_full` VALUES ('165', 'Turkey', 'A', '15', 'Core sand', 'Core sand', 'EN', 'Material', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('166', 'Turkey', 'A', '15', 'Paper, cardboard, wood', 'Paper, cardboard, wood', 'EN', 'Material', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('167', 'Turkey', 'A', '15', 'Foundry sand, furnace slag', 'Foundry sand, furnace slag', 'EN', 'Material', '99 99 99', 'Want', null, null, null, 'Get contact to Ç?MSA\nK?vanç Makine ve Deniz Döküm A.?: Can be used by analising regards of contents molting silicium,organic particules etc…');
INSERT INTO `workshop_items2_full` VALUES ('168', 'Turkey', 'A', '15', 'Pallets , big bags , sacks', 'Pallets , big bags , sacks', 'EN', 'Material', '15 01 06', 'Want', null, null, null, 'Sabiha Gökçen MTAL: Providing of education to Public and Private sector institutions.\n?lksem Mühendislik: Management Education');
INSERT INTO `workshop_items2_full` VALUES ('169', 'Turkey', 'A', '15', 'Grinding and sanding dust', 'Grinding and sanding dust', 'EN', 'Tools', '99 99 99', 'Want', null, null, null, 'Co-operation for prepartion and implementation of projects in frame of State and Private sector,national and international basis.');
INSERT INTO `workshop_items2_full` VALUES ('170', 'Turkey', 'A', '15', 'Sewage sludge', 'Sewage sludge', 'EN', 'Material', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('171', 'Turkey', 'A', '15', 'Deformed plastic bags', 'Deformed plastic bags', 'EN', 'Material', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('172', 'Turkey', 'A', '15', 'G4 type of non-hazardous air filter', 'G4 type of non-hazardous air filter', 'EN', 'Material', '99 99 99', 'Want', null, null, null, 'Ekolojik Enerji: For use in energy production and disposal facilities. Up to 60,000 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('173', 'Turkey', 'A', '15', 'Paint sludge', 'Paint sludge', 'EN', 'Material', '99 99 99', 'Want', null, null, null, 'And qualified hazardous waste (cans, drums, barrels, oil contaminated materials, metal) recovery and market launch is carried out. 4000 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('174', 'Turkey', 'A', '15', 'Rack and cutting oil waste:  can be as intermadiate storage .', 'Rack and cutting oil waste:  can be as intermadiate storage .', 'EN', 'Material', '99 99 99', 'Want', null, null, null, 'Çimsa: Alternative fuel for use in cement production. Up to 1500-2000 tonnes / year');
INSERT INTO `workshop_items2_full` VALUES ('175', 'Turkey', 'A', '15', 'Industrial Waste Oil', 'Industrial Waste Oil', 'EN', 'Material', '19 12 12', 'Want', null, null, null, 'And qualified hazardous waste (cans, drums, barrels, oil contaminated materials, metal) recovery and market launch is carried out. 4000 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('176', 'Turkey', 'A', '15', 'Cutting fluids', 'Cutting fluids', 'EN', 'Material', '19 12 12', 'Want', null, null, null, 'According to the office arising from recycling packaging, glass , bottles, paper waste');
INSERT INTO `workshop_items2_full` VALUES ('177', 'Turkey', 'A', '15', 'Waste transport services', 'Waste transport services', 'EN', 'Service', '19 12 12', 'Want', null, null, null, 'Consultancy in obtaining permits and licenses for industrial enterprises');
INSERT INTO `workshop_items2_full` VALUES ('178', 'Turkey', 'A', '15', 'Industrial packaging recycling service', 'Industrial packaging recycling service', 'EN', 'Service', '15 01 99', 'Want', null, null, null, 'In particular, the service provided to the transport of hazardous waste.');
INSERT INTO `workshop_items2_full` VALUES ('179', 'Turkey', 'A', '16', 'Metal (brass), Aluminium - copper etc. Plastic derivatives: Daily, come in varying quantities , the metal waste can be used in the foundry industry .', 'Metal (brass), Aluminium - copper etc. Plastic derivatives: Daily, come in varying quantities , the metal waste can be used in the foundry industry .', 'EN', 'Material', '17 04 01', 'Have', null, null, null, 'It is carried out in all kinds of environmental measurement and analysis laboratories (accredited Izmit )');
INSERT INTO `workshop_items2_full` VALUES ('180', 'Turkey', 'A', '16', 'Waste PP- PE-PP', 'Waste PP- PE-PP', 'EN', 'Material', '17 02 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('181', 'Turkey', 'A', '16', 'Non-hazardous metal waste , wood chips', 'Non-hazardous metal waste , wood chips', 'EN', 'Material', '17 02 01', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('182', 'Turkey', 'A', '17', 'Roof tile waste: Waste resulting from production wastage. Companies can receive weekly or monthly period. Fire -resistant, this product is not harmful to health or used in road construction in the running surface.', 'Roof tile waste: Waste resulting from production wastage. Companies can receive weekly or monthly period. Fire -resistant, this product is not harmful to health or used in road construction in the running surface.', 'EN', 'Material', '17 01 03', 'Have', '60000', 'tons', 'per year', 'Alternative fuel for use in cement production. 60,000 tons / year can be taken up .');
INSERT INTO `workshop_items2_full` VALUES ('183', 'Turkey', 'A', '17', 'Hazardous / non-hazardous waste', 'Hazardous / non-hazardous waste', 'EN', 'Material', '17 09 03', 'Have', null, null, null, 'The separation of industrial and domestic waste, used to convert energy from organic waste. 300 tons / day of decomposed.');
INSERT INTO `workshop_items2_full` VALUES ('184', 'Turkey', 'A', '17', 'Assessable mixed packaging waste', 'Assessable mixed packaging waste', 'EN', 'Material', '15 01 06', 'Have', null, null, null, 'Iron, chromium, manganese content lower materials\nSand , quartzite, limestone , marble, iron, aluminum, sodium and potassium content is high materials.');
INSERT INTO `workshop_items2_full` VALUES ('185', 'Turkey', 'A', '17', 'Electronic waste: Endel A.?. They are working on as a collection', 'Electronic waste: Endel A.?. They are working on as a collection', 'EN', 'Material', '17 09 04', 'Have', null, null, null, 'Alternative fuel for use in cement production. 1500-2000 tons / year can be taken up .');
INSERT INTO `workshop_items2_full` VALUES ('186', 'Turkey', 'A', '17', 'Hazardous recyclable waste: Endel A.?. As work on this issue.', 'Hazardous recyclable waste: Endel A.?. As work on this issue.', 'EN', 'Material', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('187', 'Turkey', 'A', '17', 'Clay -containing waste: Shipping costs and breaking are met by Endel .', 'Clay -containing waste: Shipping costs and breaking are met by Endel .', 'EN', 'Material', '17 09 04', 'Have', null, null, null, '250 tons / month');
INSERT INTO `workshop_items2_full` VALUES ('188', 'Turkey', 'A', '17', 'Aggregate: Concrete tiles can be taken as one of the main components.', 'Aggregate: Concrete tiles can be taken as one of the main components.', 'EN', 'Material', '17 09 04', 'Want', null, null, null, 'Marble quarry is operating rubble waste ( limestone ) . It emerges from the Yuce Mining ( Bilecik) . Cement may be the raw material . 500,000 tons / year .');
INSERT INTO `workshop_items2_full` VALUES ('189', 'Turkey', 'A', '17', 'Iron and steel slag: Concrete tiles can be taken as one of the main components.', 'Iron and steel slag: Concrete tiles can be taken as one of the main components.', 'EN', 'Material', '17 09 04', 'Want', null, null, null, 'Raw materials for cement production , contribution and analysis of all kinds of materials can be used as fuel.');
INSERT INTO `workshop_items2_full` VALUES ('190', 'Turkey', 'A', '17', 'Core sand: Can be used as aggregate.', 'Core sand: Can be used as aggregate.', 'EN', 'Material', '17 09 04', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('191', 'Turkey', 'A', '17', 'wood Pellet', 'wood Pellet', 'EN', 'Tools', '17 09 04', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('192', 'Turkey', 'A', '17', 'consultancy', 'consultancy', 'EN', 'Service', '17 09 04', 'Want', null, null, null, 'Partnerships or joint projects can be made with other companies and institutions');
INSERT INTO `workshop_items2_full` VALUES ('193', 'Turkey', 'A', '17', 'Testing / Analysis', 'Testing / Analysis', 'EN', 'Service', '17 09 04', 'Want', null, null, null, 'ES Anka Havac?l?k: KOSGEB with new companies in the aviation sector , he wants to get support from organizations such as BEBKA . Quality certification, can receive advice on issues such as machinery and equipment of the company.\nAk Geri Dönü?üm: Consultancy relating to financial support.\nAk Oluklu Ambalaj: Corporate Governance, HR and R & D support');
INSERT INTO `workshop_items2_full` VALUES ('194', 'Turkey', 'A', '18', 'Service: Project coordination and execution services provide competence in regional projects', 'Service: Project coordination and execution services provide competence in regional projects', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Production and management processes');
INSERT INTO `workshop_items2_full` VALUES ('195', 'Turkey', 'A', '18', 'Service: Possibility to access EU funds for international partners and funds to develop the process. Industrial Symbiosis scope to connect with overseas companies.', 'Service: Possibility to access EU funds for international partners and funds to develop the process. Industrial Symbiosis scope to connect with overseas companies.', 'EN', 'Service', '16 02 16', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('196', 'Turkey', 'A', '18', 'Energy , human resources and project execution', 'Energy , human resources and project execution', 'EN', 'Energy', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('197', 'Turkey', 'A', '18', 'Consulting and investment support', 'Consulting and investment support', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'KOSGEB with new companies in the aviation sector , he wants to get support from organizations such as BEBKA . Quality certification, can receive advice on issues such as machinery and equipment of the company.');
INSERT INTO `workshop_items2_full` VALUES ('198', 'Turkey', 'A', '18', 'Training and Consulting', 'Training and Consulting', 'EN', 'Service', '12 01 10', 'Have', null, null, null, 'OZI staff in a kindergarten or nursery children can go . Hours to prevent losses will facilitate the accessibility to children.');
INSERT INTO `workshop_items2_full` VALUES ('199', 'Turkey', 'A', '18', 'Training', 'Training', 'EN', 'Service', '12 01 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('200', 'Turkey', 'A', '19', 'Chip (Aluminium, Iron, Bronze ): Wastes from the manufacturing process.', 'Chip (Aluminium, Iron, Bronze ): Wastes from the manufacturing process.', 'EN', 'Material', '12 01 03', 'Have', null, null, null, 'Sabiha Gökçen MTAL: Submission services to public and private vocational training institutions.\n?lksem Mühendislik: Project writing, quality management and executive training.');
INSERT INTO `workshop_items2_full` VALUES ('201', 'Turkey', 'A', '19', 'Consulting and investment support: KOSGEB with new companies in the aviation sector , he wants to get support from organizations such as BEBKA . Quality certification, can receive advice on issues such as machinery and equipment of the company.', 'Consulting and investment support: KOSGEB with new companies in the aviation sector , he wants to get support from organizations such as BEBKA . Quality certification, can receive advice on issues such as machinery and equipment of the company.', 'EN', 'Service', '12 01 03', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('202', 'Turkey', 'A', '19', 'Education or vocational training welders for welding manufacturing', 'Education or vocational training welders for welding manufacturing', 'EN', 'Service', '12 01 03', 'Want', null, null, null, 'Big bags and bags for use in food products.');
INSERT INTO `workshop_items2_full` VALUES ('203', 'Turkey', 'A', '19', 'Destructive / Nondestructive Laboratory Services', 'Destructive / Nondestructive Laboratory Services', 'EN', 'Service', '12 01 03', 'Want', null, null, null, 'Eski?ehir Industrial Energy, this service is already from ESÇEV Engineering.');
INSERT INTO `workshop_items2_full` VALUES ('204', 'Turkey', 'A', '19', 'Education', 'Education', 'EN', 'Service', '12 01 03', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('205', 'Turkey', 'A', '19', 'Packing certification services', 'Packing certification services', 'EN', 'Service', '12 01 03', 'Want', null, null, null, 'Seranit: Readily it gives .\nTÜLOMSA?: Barrels can given.\nEndel A.?: It works on the subject.');
INSERT INTO `workshop_items2_full` VALUES ('206', 'Turkey', 'A', '20', 'Waste water: Basedon  Eskisehir SKKY Table.19 OSB waste water treatment .', 'Waste water: Basedon  Eskisehir SKKY Table.19 OSB waste water treatment .', 'EN', 'Material', '19 08 99', 'Have', '16500', 'm3', 'per day', 'ES Anka Havac?l?k: Wastes from the manufacturing process .');
INSERT INTO `workshop_items2_full` VALUES ('207', 'Turkey', 'A', '20', 'consultancy', 'consultancy', 'EN', 'Service', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('208', 'Turkey', 'A', '20', 'Accredited wastewater laboratory: Wastewater analysis & sampling', 'Accredited wastewater laboratory: Wastewater analysis & sampling', 'EN', 'Service', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('209', 'Turkey', 'A', '20', 'Odor removal of waste water: Eski?ehir OZI sourced from ATT', 'Odor removal of waste water: Eski?ehir OZI sourced from ATT', 'EN', 'Service', '02 01 99', 'Want', null, null, null, 'Already going to Istanbul. Analysis already done (for dangerous) , may used in the elevator industry. Core sand composition was sio? . 4160 kg / year .');
INSERT INTO `workshop_items2_full` VALUES ('210', 'Turkey', 'A', '20', 'Packing certification services', 'Packing certification services', 'EN', 'Service', '15 01 99', 'Want', null, null, null, 'Deformed, plastic bags , which are unable to be used. 10 kg / month .');
INSERT INTO `workshop_items2_full` VALUES ('211', 'Turkey', 'A', '21', 'Pallets , big bags , sacks: ESÇEV and  Seranit collaborate and perform together.', 'Pallets , big bags , sacks: ESÇEV and  Seranit collaborate and perform together.', 'EN', 'Tools', '15 01 05', 'Have', null, null, null, 'Hazardous metal waste thrown the case out of steam and gas turbine systems used');
INSERT INTO `workshop_items2_full` VALUES ('212', 'Turkey', 'A', '21', 'consultancy: Consultancy in obtaining permits and licenses for industrial enterprises', 'consultancy: Consultancy in obtaining permits and licenses for industrial enterprises', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Receive services related to recycling technologies, waste collection center for the establishment of a system of waste that may be collected from households and installation of this system.');
INSERT INTO `workshop_items2_full` VALUES ('213', 'Turkey', 'A', '21', 'Environmental measures: It is carried out in all kinds of environmental measurement and analysis laboratories ( accredited Izmit )', 'Environmental measures: It is carried out in all kinds of environmental measurement and analysis laboratories ( accredited Izmit )', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Waste recycling facilities for paper mills or considered as intermediate products or raw materials. 30 tons / day .');
INSERT INTO `workshop_items2_full` VALUES ('214', 'Turkey', 'A', '21', 'Hazardous recyclable waste: qualified and  hazardous waste ( cans , drums , barrels, oil contaminated materials , metal ) recovery and market launch is carried out.', 'Hazardous recyclable waste: qualified and  hazardous waste ( cans , drums , barrels, oil contaminated materials , metal ) recovery and market launch is carried out.', 'EN', 'Material', '15 01 06', 'Want', '4000', 'tons', 'per year', 'Already given to Özvar Endüstriyel At?k Ambalaj.');
INSERT INTO `workshop_items2_full` VALUES ('215', 'Turkey', 'A', '21', 'Chip (Aluminium, Iron, Bronze )', 'Chip (Aluminium, Iron, Bronze )', 'EN', 'Material', '12 01 03', 'Want', null, null, null, 'Odunpazar? Belediyesi: They have to work  about it .\nEndel A.?: It works about the picking up.');
INSERT INTO `workshop_items2_full` VALUES ('216', 'Turkey', 'A', '21', 'Hazardous / Non-hazardous metal waste , wood chips', 'Hazardous / Non-hazardous metal waste , wood chips', 'EN', 'Material', '03 01 01', 'Want', null, null, null, 'Appropriate special machine to perform specific actions depending on the requirements, the process and appropriate software support them.');
INSERT INTO `workshop_items2_full` VALUES ('217', 'Turkey', 'A', '21', 'Waste copper cable', 'Waste copper cable', 'EN', 'Material', '12 01 03', 'Want', null, null, null, 'It works with the instructors.');
INSERT INTO `workshop_items2_full` VALUES ('218', 'Turkey', 'A', '21', 'Core sand', 'Core sand', 'EN', 'Material', '12 01 03', 'Want', null, null, null, 'Wastewater treatment plant on the business and investment .');
INSERT INTO `workshop_items2_full` VALUES ('219', 'Turkey', 'A', '21', 'Deformed plastic bags', 'Deformed plastic bags', 'EN', 'Material', '07 02 13', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('220', 'Turkey', 'A', '21', 'Metal waste', 'Metal waste', 'EN', 'Material', '12 01 03', 'Want', null, null, null, 'TTGV: Pine bark , which absorbs the odor removal .');
INSERT INTO `workshop_items2_full` VALUES ('221', 'Turkey', 'A', '21', 'Integrated waste ( packaging - waste oil and Electronic Waste , etc.).', 'Integrated waste ( packaging - waste oil and Electronic Waste , etc.).', 'EN', 'Material', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('222', 'Turkey', 'A', '21', 'Pressed scrap paper', 'Pressed scrap paper', 'EN', 'Material', '12 01 03', 'Want', '30', 'tons ', 'per day', 'H?zlan: Only can take the aluminum shavings');
INSERT INTO `workshop_items2_full` VALUES ('223', 'Turkey', 'A', '21', 'Chemical contaminated \ndrums / barrels', 'Chemical contaminated \ndrums / barrels', 'EN', 'Tools', '15 01 10', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('224', 'Turkey', 'A', '21', 'Electronic waste: Recovery of all types of electronic waste arising from industrial establishments and other places are carried out.', 'Electronic waste: Recovery of all types of electronic waste arising from industrial establishments and other places are carried out.', 'EN', 'Service', '12 01 03', 'Want', '2000', 'tons', 'per year', 'TÜLOMSA?: Akredite educational services for welder training ( 17024 )');
INSERT INTO `workshop_items2_full` VALUES ('225', 'Turkey', 'A', '21', 'Mechatronics and software support', 'Mechatronics and software support', 'EN', 'Service', '12 01 03', 'Want', null, null, null, 'Mechanical - endete can be served in the examination. Accredited also to Destructive will be accredited in the future.');
INSERT INTO `workshop_items2_full` VALUES ('226', 'Turkey', 'A', '22', 'Food and textile: Firmalar ad?na', 'Food and textile: Firmalar ad?na', 'EN', 'Material', '99 99 99', 'Have', null, null, null, 'Project writing, quality management and executive training');
INSERT INTO `workshop_items2_full` VALUES ('227', 'Turkey', 'A', '22', 'Consulting and investment support', 'Consulting and investment support', 'EN', 'Service', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('228', 'Turkey', 'A', '22', 'Nursery / Kindergarten', 'Nursery / Kindergarten', 'EN', 'Service', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('229', 'Turkey', 'A', '22', 'Energy , human resources and project execution: Partnerships or joint projects can be made with other companies and institutions', 'Energy , human resources and project execution: Partnerships or joint projects can be made with other companies and institutions', 'EN', 'Energy', '99 99 99', 'Want', null, null, null, 'For use in energy production in disposal facilities . 60.00 tons / year capacity.');
INSERT INTO `workshop_items2_full` VALUES ('230', 'Turkey', 'A', '22', 'Training Service', 'Training Service', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'The least amount of packaging waste from businesses assessed . Can be collected in the  Odunpazari region. 1000 tons / month .');
INSERT INTO `workshop_items2_full` VALUES ('231', 'Turkey', 'A', '22', 'Packing certification services', 'Packing certification services', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Recovery of all types of electronic waste arising from industrial establishments and other places are carried out. 2000 tons / year capacity is running.');
INSERT INTO `workshop_items2_full` VALUES ('232', 'Turkey', 'A', '23', 'Construction and demolition waste: Residual waste from construction waste recycling plants that occur in the city.', 'Construction and demolition waste: Residual waste from construction waste recycling plants that occur in the city.', 'EN', 'Material', '17 09 04', 'Have', '500', 'tons', 'per day', 'Qualified hazardous waste ( cans , drums , barrels, oil contaminated materials , metal ) recovery and market launch is carried out. 4000 tons / year capacity convenience.');
INSERT INTO `workshop_items2_full` VALUES ('233', 'Turkey', 'A', '23', 'Home thrash', 'Home thrash', 'EN', 'Material', '20 03 01', 'Have', '800', 'tons', 'per day', 'Alternative raw materials for use in cement production. 200,000-300,000 tonnes / year capacity purchase.');
INSERT INTO `workshop_items2_full` VALUES ('234', 'Turkey', 'A', '23', 'Solids', 'Solids', 'EN', 'Material', '20 03 01', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('235', 'Turkey', 'A', '23', 'mixed Waste', 'mixed Waste', 'EN', 'Material', '20 03 01', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('236', 'Turkey', 'A', '23', 'Natural mineral waste', 'Natural mineral waste', 'EN', 'Material', '06 13 99', 'Have', null, null, null, 'Already going to Istanbul. Analysis are (dangerous) , used in the elevator industry. Core sand composition was sio? . 4160 kg / year .');
INSERT INTO `workshop_items2_full` VALUES ('237', 'Turkey', 'A', '23', 'waste oil', 'waste oil', 'EN', 'Material', '12 01 10', 'Have', null, null, null, 'Stowage and shipment of materials can be produced for use in industrial companies.');
INSERT INTO `workshop_items2_full` VALUES ('238', 'Turkey', 'A', '23', 'Advice on waste management', 'Advice on waste management', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Companies to be given advice on waste management and system.');
INSERT INTO `workshop_items2_full` VALUES ('239', 'Turkey', 'A', '23', 'Home thrash', 'Home thrash', 'EN', 'Material', '20 03 01', 'Want', null, null, null, 'Chemical and physical analysis. Mineralogical analysis ( XRD and XRF ) service. Accredited ceramic final product testing. Boron analysis services provided.');
INSERT INTO `workshop_items2_full` VALUES ('240', 'Turkey', 'A', '23', 'Debris waste', 'Debris waste', 'EN', 'Material', '17 09 03', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('241', 'Turkey', 'A', '23', 'Lab', 'Lab', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Ensuring the recovery of waste polyurethane and polypropylene plastic material used to obtain granular raw material and industrial bags is obtained.');
INSERT INTO `workshop_items2_full` VALUES ('242', 'Turkey', 'A', '24', 'Metal waste: Hazardous metal waste thrown the case out of steam and gas turbine systems used', 'Metal waste: Hazardous metal waste thrown the case out of steam and gas turbine systems used', 'EN', 'Material', '10 01 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('243', 'Turkey', 'A', '24', 'G4 type of hazardous air filter: Filtre of gas turbine air intake', 'G4 type of hazardous air filter: Filtre of gas turbine air intake', 'EN', 'Material', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('244', 'Turkey', 'A', '24', 'Hazardous / non-hazardous waste', 'Hazardous / non-hazardous waste', 'EN', 'Material', '99 99 99', 'Have', null, null, null, 'Recovery of all types of electronic waste arising from industrial establishments and other places are carried out. 20 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('245', 'Turkey', 'A', '24', 'Hazardous recyclable waste', 'Hazardous recyclable waste', 'EN', 'Material', '12 01 03', 'Have', null, null, null, 'Big bags and bags for use in food products.');
INSERT INTO `workshop_items2_full` VALUES ('246', 'Turkey', 'A', '24', 'Industrial waste oil: Oils used in steam and gas turbine units', 'Industrial waste oil: Oils used in steam and gas turbine units', 'EN', 'Material', '19 12 12', 'Have', null, null, null, 'Paper , plastic, pallets , glass, metal , composites can be used as raw material in wood packaging waste facility . 10,000 tons / year .');
INSERT INTO `workshop_items2_full` VALUES ('247', 'Turkey', 'A', '24', 'Hazardous recyclable waste', 'Hazardous recyclable waste', 'EN', 'Material', '15 01 10', 'Want', null, null, null, 'To prepare for re-use');
INSERT INTO `workshop_items2_full` VALUES ('248', 'Turkey', 'A', '24', 'Hazardous waste', 'Hazardous waste', 'EN', 'Material', '20 03 01', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('249', 'Turkey', 'A', '24', 'Consultancy', 'Consultancy', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Being non-hazardous nature of the content consists of silica sand + bentonite + coal dust. 20 tons / month .');
INSERT INTO `workshop_items2_full` VALUES ('250', 'Turkey', 'A', '24', 'Waste transport services', 'Waste transport services', 'EN', 'Service', '99 99 99', 'Want', null, null, null, '1 tonne / year');
INSERT INTO `workshop_items2_full` VALUES ('251', 'Turkey', 'A', '24', 'Environmental measures', 'Environmental measures', 'EN', 'Service', '99 99 99', 'Want', null, null, null, '40 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('252', 'Turkey', 'A', '25', 'Foundry sand: The waste of factories working in organized industrial zone. These wastes are characterized as hazardous and/or non-hazardous according to analysis', 'Foundry sand: The waste of factories working in organized industrial zone. These wastes are characterized as hazardous and/or non-hazardous according to analysis', 'EN', 'Material', '99 99 99', 'Have', null, null, null, 'Aksoylu Trayler: 1 tonne / year\nTÜLOMSA?:Wood packaging waste .');
INSERT INTO `workshop_items2_full` VALUES ('253', 'Turkey', 'A', '25', 'Sewage sludge: %80 dry. 2000-2400 kcal calorific value', 'Sewage sludge: %80 dry. 2000-2400 kcal calorific value', 'EN', 'Material', '17 01 07', 'Have', '35', 'tons', 'per day', '200 pcs / year');
INSERT INTO `workshop_items2_full` VALUES ('254', 'Turkey', 'A', '25', 'Study Metal coating industry, chrome wastes use in ceramic industry: Chrome precipitation unit is needed', 'Study Metal coating industry, chrome wastes use in ceramic industry: Chrome precipitation unit is needed', 'EN', 'Material', '10 02 01', 'Have', null, null, null, 'Creating the risk of fire scattered&non scattered waste paper can be delivered to the paper mill scrap of paper . 16 tons / month .');
INSERT INTO `workshop_items2_full` VALUES ('255', 'Turkey', 'A', '26', 'Expertise-laboratory services: Consultancy on recycling of metal waste and its use in ceramics industry Consultancy on production of methane from animal waste', 'Expertise-laboratory services: Consultancy on recycling of metal waste and its use in ceramics industry Consultancy on production of methane from animal waste', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Fruit peel, fruit syrups . 1 tonne / week .');
INSERT INTO `workshop_items2_full` VALUES ('256', 'Turkey', 'A', '26', 'Consultancy: About Metallurgy', 'Consultancy: About Metallurgy', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Municipalities receive support from the respective companies for the establishment of the system for their own waste and electronic waste collected from households and to start the recovery process');
INSERT INTO `workshop_items2_full` VALUES ('257', 'Turkey', 'A', '27', 'Sewage sludge: The waste from treatment of process water in dough production. 50% dry', 'Sewage sludge: The waste from treatment of process water in dough production. 50% dry', 'EN', 'Material', '02 06 99', 'Have', '15', 'm3', 'per day', 'Continuous waste from the ceramics factory . Inert , 17% humidity . 1000 tons / month .');
INSERT INTO `workshop_items2_full` VALUES ('258', 'Turkey', 'A', '28', 'Grinding and sandblasting dust: Industrial non-hazardous waste', 'Grinding and sandblasting dust: Industrial non-hazardous waste', 'EN', 'Material', '99 99 99', 'Have', '1', 'tons', 'per month', null);
INSERT INTO `workshop_items2_full` VALUES ('259', 'Turkey', 'A', '28', 'Household waste: Can be esed in the production of biogas and compost', 'Household waste: Can be esed in the production of biogas and compost', 'EN', 'Material', '21 01 08', 'Have', '30', 'tons', 'per month', '2850 kg / year (mixed)');
INSERT INTO `workshop_items2_full` VALUES ('260', 'Turkey', 'A', '28', '\nRefractory materials', 'Refractory materials', 'EN', 'Material', '16 11 06', 'Have', null, null, null, 'Already going to Istanbul. Analysis are (dangerous) , used in the elevator industry. Core sand composition was sio? . 4160 kg / year .');
INSERT INTO `workshop_items2_full` VALUES ('261', 'Turkey', 'A', '28', 'Fertilizer waste', 'Fertilizer waste', 'EN', 'Material', '06 10 02', 'Have', null, null, null, '800-1000 kg / year');
INSERT INTO `workshop_items2_full` VALUES ('262', 'Turkey', 'A', '28', 'Mixed waste: The separation of industrial and household wastes can be used for turning the organic wastes into energy. This issue should be taken into consideration by organized industrial zone', 'Mixed waste: The separation of industrial and household wastes can be used for turning the organic wastes into energy. This issue should be taken into consideration by organized industrial zone', 'EN', 'Material', '16 02 16', 'Have', null, null, null, 'Hazardous , there hve analyzes.');
INSERT INTO `workshop_items2_full` VALUES ('263', 'Turkey', 'A', '28', '\nGrease trap waste: It is the sedimentary section of the refectory wastewater deposited on top after passing the grease trap. Can be used in soap and chemical industries', '\nGrease trap waste: It is the sedimentary section of the refectory wastewater deposited on top after passing the grease trap. Can be used in soap and chemical industries', 'EN', 'Material', '13 05 08', 'Have', '2', 'tons', 'per month', 'Big bags and bags for use in food products.');
INSERT INTO `workshop_items2_full` VALUES ('264', 'Turkey', 'A', '28', '\nWaste and hazardous waste management', '\nWaste and hazardous waste management', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Industrial non-hazardous waste . 1 tonne / month (variable)');
INSERT INTO `workshop_items2_full` VALUES ('265', 'Turkey', 'A', '29', 'PVC foil: The waste of production process', 'PVC foil: The waste of production process', 'EN', 'Material', '03 01 99', 'Have', '50', 'kg', 'per day', 'ET? G?da: Pulp production is due to the treatment of the process water . 50% dryness.15 m³ / day.\nEski?ehir OZI: 80% of dryness. There are 2000-2400 kcal thermal value. 30 tons / day .');
INSERT INTO `workshop_items2_full` VALUES ('266', 'Turkey', 'A', '29', 'Laminate: Resin-impregnated paper. Availability in various colors and sizes 0.5 / 0.7 micron pieces', 'Laminate: Resin-impregnated paper. Availability in various colors and sizes 0.5 / 0.7 micron pieces', 'EN', 'Material', '03 01 05', 'Have', null, null, null, 'Deformed, which can not be used in plastic bags.10 kg / month .');
INSERT INTO `workshop_items2_full` VALUES ('267', 'Turkey', 'A', '29', 'Orman ürünleri üretim art???, ah?ap at???', 'Orman ürünleri üretim art???, ah?ap at???', 'EN', 'Unknown', '99 99 99', 'Have', null, null, null, 'Gas turbine intake air filter');
INSERT INTO `workshop_items2_full` VALUES ('268', 'Turkey', 'A', '29', 'Glue sludge: Glue cabinet is caused by the water curtain pool in production process. Analysis are available and non-hazardous', 'Glue sludge: Glue cabinet is caused by the water curtain pool in production process. Analysis are available and non-hazardous', 'EN', 'Material', '03 01 99', 'Have', '25', 'kg', null, 'Wet coating process resulting paint sludge . 9 tonnes / year.');
INSERT INTO `workshop_items2_full` VALUES ('269', 'Turkey', 'A', '30', 'Aluminum shavings', 'Aluminum shavings', 'EN', 'Material', '12 01 03', 'Have', null, null, null, 'Containering of waste oil disposal or use of the benches at work again. 10 barrels / year.');
INSERT INTO `workshop_items2_full` VALUES ('270', 'Turkey', 'A', '30', 'Core sand', 'Core sand', 'EN', 'Material', '12 01 03', 'Want', null, null, null, 'Oils used in steam and gas turbine units');
INSERT INTO `workshop_items2_full` VALUES ('271', 'Turkey', 'A', '30', 'Waste machine oils: Licence needed', 'Waste machine oils: Licence needed', 'EN', 'Material', '12 01 03', 'Want', null, null, null, 'Hazardous waste is set . It consists of tramp oil and coolant . 1 bbl / month .');
INSERT INTO `workshop_items2_full` VALUES ('272', 'Turkey', 'A', '31', 'Services / Consulting: Workforce, equipment and IT services on the organization of logistics', 'Services / Consulting: Workforce, equipment and IT services on the organization of logistics', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'In particular, the service provided to the transport of hazardous waste.');
INSERT INTO `workshop_items2_full` VALUES ('273', 'Turkey', 'A', '31', 'Transportation', 'Transportation', 'EN', 'Service', '12 01 10', 'Have', null, null, null, 'Metal - plastic barrels , drums , IBC tank');
INSERT INTO `workshop_items2_full` VALUES ('274', 'Turkey', 'A', '31', 'Pallets, big bags, sacks: Big bags and bags for use in food products', 'Pallets, big bags, sacks: Big bags and bags for use in food products', 'EN', 'Tools', '15 01 06', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('275', 'Turkey', 'A', '31', 'Deformed plastic bags', 'Deformed plastic bags', 'EN', 'Material', '15 01 02', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('276', 'Turkey', 'A', '32', 'Plaster mold waste: The waste of a customer firm', 'Plaster mold waste: The waste of a customer firm', 'EN', 'Material', '17 09 04', 'Have', '1000', 'tons', 'per year', 'Director of the municipal park with gardens and rural services provided resources for use in the office.');
INSERT INTO `workshop_items2_full` VALUES ('277', 'Turkey', 'A', '32', 'Raw materials and various chemicals: Clay group minerals, feldspar, alumina etc.', 'Raw materials and various chemicals: Clay group minerals, feldspar, alumina etc.', 'EN', 'Material', '17 09 04', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('278', 'Turkey', 'A', '32', 'Laboratory services: The use of devices such as jet mill, ball mill, high temperature ovens, pH meter', 'Laboratory services: The use of devices such as jet mill, ball mill, high temperature ovens, pH meter', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Of compressed sawdust and various chemicals may be pressed and given to the forest villagers as firewood/combustable . 10,000 tons / year capacity.');
INSERT INTO `workshop_items2_full` VALUES ('279', 'Turkey', 'A', '32', 'Training: Project writing, quality management and executive trainings', 'Training: Project writing, quality management and executive trainings', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'At the planning stage , to be used as biofuels.');
INSERT INTO `workshop_items2_full` VALUES ('280', 'Turkey', 'A', '32', 'Consulting and investment support', 'Consulting and investment support', 'EN', 'Service', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('281', 'Turkey', 'A', '32', 'Energy, human resources and project management', 'Energy, human resources and project management', 'EN', 'Energy', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('282', 'Turkey', 'A', '32', 'Ceramic raw materials and production waste', 'Ceramic raw materials and production waste', 'EN', 'Material', '99 99 99', 'Want', null, null, null, '0.5 tonnes / year');
INSERT INTO `workshop_items2_full` VALUES ('283', 'Turkey', 'A', '32', 'Laminate: Small amounts to use in R&D studies', 'Laminate: Small amounts to use in R&D studies', 'EN', 'Material', '07 02 13', 'Want', null, null, null, 'Every day, come in varying quantities , the metal waste can be used in the foundry industry.');
INSERT INTO `workshop_items2_full` VALUES ('284', 'Turkey', 'A', '32', 'Filter press sludge', 'Filter press sludge', 'EN', 'Material', '17 01 07', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('285', 'Turkey', 'A', '32', 'Sawmill dust and shavings', 'Sawmill dust and shavings', 'EN', 'Material', '15 01 01', 'Want', null, null, null, 'Trimmed , broken scrap plastic');
INSERT INTO `workshop_items2_full` VALUES ('286', 'Turkey', 'A', '32', 'Marketing and sales consulting: Market analysis, marketing and sales strategies', 'Marketing and sales consulting: Market analysis, marketing and sales strategies', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'In particular, the service provided to the transport of hazardous waste.');
INSERT INTO `workshop_items2_full` VALUES ('287', 'Turkey', 'A', '32', 'Mechatronics and software support: On behalf of the consultant company', 'Mechatronics and software support: On behalf of the consultant company', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Ak Geri Dönü?üm: Used as raw materials taking into big bags.');
INSERT INTO `workshop_items2_full` VALUES ('288', 'Turkey', 'A', '32', '\nTesting / Analysis', '\nTesting / Analysis', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('289', 'Turkey', 'A', '33', 'Deformed plastic bags: Deformed / unusable plastic bags', 'Deformed plastic bags: Deformed / unusable plastic bags', 'EN', 'Material', '99 99 99', 'Have', '10', 'kg', 'per month', 'Foundry sand silica and so on. analyzed for containing organic compounds can be used continuously.');
INSERT INTO `workshop_items2_full` VALUES ('290', 'Turkey', 'A', '33', 'House waste / food waste', 'House waste / food waste', 'EN', 'Material', '02 02 99', 'Have', null, null, null, 'For use in road construction work for Municipality . Iron slag out of the sea can be used as road construction material.');
INSERT INTO `workshop_items2_full` VALUES ('291', 'Turkey', 'A', '34', 'Contaminated waste oil\nBoron oil\nHydraulic oil', 'Contaminated waste oil\nBoron oil\nHydraulic oil', 'EN', 'Material', '12 01 06', 'Have', '2850', 'kg', 'per year', 'Considered the least amount of packaging waste collected from factories within the boundaries of Odunpazari . 1000 tons / month');
INSERT INTO `workshop_items2_full` VALUES ('292', 'Turkey', 'A', '34', 'Core sand: Already going to Istanbul. Analysis (hazardous) are available, can be used in elevator/lift industry. Core sand composition is SiO?', 'Core sand: Already going to Istanbul. Analysis (hazardous) are available, can be used in elevator/lift industry. Core sand composition is SiO?', 'EN', 'Material', '12 01 99', 'Have', '4160', 'kg', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('293', 'Turkey', 'A', '34', 'Sprue', 'Sprue', 'EN', 'Tools', '12 01 03', 'Have', '1', 'tons', 'per month', 'To investigate .');
INSERT INTO `workshop_items2_full` VALUES ('294', 'Turkey', 'A', '34', '\nSeedlings soil: Foundry sand can be used as a resource after analyzing the containment of silicon and organic compounds', '\nSeedlings soil: Foundry sand can be used as a resource after analyzing the containment of silicon and organic compounds', 'EN', 'Material', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('295', 'Turkey', 'A', '34', '\nPaper, cardboard, wood', '\nPaper, cardboard, wood', 'EN', 'Material', '15 01 01', 'Have', '900', 'kg', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('296', 'Turkey', 'A', '34', 'Asphalt and other road-building materials: Slags from plants can be used in road construction', 'Asphalt and other road-building materials: Slags from plants can be used in road construction', 'EN', 'Material', '10 09 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('297', 'Turkey', 'A', '34', 'Wood, timber, fertilizer: Slag and foundry sand can be used', 'Wood, timber, fertilizer: Slag and foundry sand can be used', 'EN', 'Material', '03 01 01', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('298', 'Turkey', 'A', '34', 'Iron and steel slag', 'Iron and steel slag', 'EN', 'Material', '17 09 04', 'Have', null, null, null, 'Ak Geri Dönü?üm; There are as clippings.');
INSERT INTO `workshop_items2_full` VALUES ('299', 'Turkey', 'A', '34', 'Aggregate (agrega)', 'Aggregate (agrega)', 'EN', 'Material', '17 09 04', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('300', 'Turkey', 'A', '34', 'Foundry sand, furnace slag: Hazardous, analyzes available', 'Foundry sand, furnace slag: Hazardous, analyzes available', 'EN', 'Material', '10 09 99', 'Have', '100', 'tons', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('301', 'Turkey', 'A', '34', 'Metal scrap: Need for steel scrap material as raw material in melting process', 'Metal scrap: Need for steel scrap material as raw material in melting process', 'EN', 'Material', '12 01 01', 'Want', '30', 'tons', 'per month', 'Stowage and shipment of materials can be produced for use in industrial companies.');
INSERT INTO `workshop_items2_full` VALUES ('302', 'Turkey', 'A', '34', 'DKP-Scrap Steel', 'DKP-Scrap Steel', 'EN', 'Material', '12 01 01', 'Want', null, null, null, 'Mechanical - endete can be served in the examination. Destructive will be accredited for the forthcoming period.');
INSERT INTO `workshop_items2_full` VALUES ('303', 'Turkey', 'A', '34', 'Metal (brass), Aluminium - copper etc. Plastic derivatives', 'Metal (brass), Aluminium - copper etc. Plastic derivatives', 'EN', 'Material', '12 01 05', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('304', 'Turkey', 'A', '34', '\nChip (Aluminium, Iron, Bronze)', '\nChip (Aluminium, Iron, Bronze)', 'EN', 'Material', '12 01 03', 'Want', null, null, null, 'ESO: Ekobord new generation of plate ( Bilecik) are the resources that it can get . 60 tons / month');
INSERT INTO `workshop_items2_full` VALUES ('305', 'Turkey', 'A', '35', '\nSpoilage / waste paper', '\nSpoilage / waste paper', 'EN', 'Material', '15 01 01', 'Have', '750', 'kg', 'per month', 'Refractory materials coming from the quarries used ( stone wool, lining etc.) 1 tonne / month');
INSERT INTO `workshop_items2_full` VALUES ('306', 'Turkey', 'A', '35', 'Kraft cellulose such as cellulose, cardboard and cement bags', 'Kraft cellulose such as cellulose, cardboard and cement bags', 'EN', 'Material', '15 01 01', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('307', 'Turkey', 'A', '35', 'Sawmill dust and shavings', 'Sawmill dust and shavings', 'EN', 'Material', '15 01 01', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('308', 'Turkey', 'A', '36', 'Packaging waste: Can provide software support', 'Packaging waste: Can provide software support', 'EN', 'Unknown', '12 01 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('309', 'Turkey', 'A', '36', '\nConsultancy: Project management, consulting on software and hardware issues', '\nConsultancy: Project management, consulting on software and hardware issues', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Endel A.?.:Shipping costs and breaking are met by Endel . Roof tile wastes are wastes from production waste . Companies can receive weekly or monthly period . Fire -resistant , this product is not harmful to health or used in road construction in the running surface . ~ 60,000 tons / year.\n?lksem Mühendislik: Clay group minerals , feldspar , alumina , etc .');
INSERT INTO `workshop_items2_full` VALUES ('310', 'Turkey', 'A', '36', 'Mechatronics and software support: Consultancy services on machinery, system, software, hardware and process development', 'Mechatronics and software support: Consultancy services on machinery, system, software, hardware and process development', 'EN', 'Service', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('311', 'Turkey', 'A', '36', 'Training', 'Training', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'ESART A.?.: Eski?ehir OSB arising from the WWTP . Dryness ratio of 40 %. With solar drying plant to be installed will be 80 % dryness . In this case the amount of sludge day 27 ton / day may be ~ 2200 kcal');
INSERT INTO `workshop_items2_full` VALUES ('312', 'Turkey', 'A', '36', 'Marketing', 'Marketing', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Residual waste from construction waste recycling plants that occur in the city. 500 tons / day.');
INSERT INTO `workshop_items2_full` VALUES ('313', 'Turkey', 'A', '37', 'Waste Mineral Oil: From the manufacturing of automotive parts', 'Waste Mineral Oil: From the manufacturing of automotive parts', 'EN', 'Material', '12 01 10', 'Have', '1', 'tons', 'per year', 'Baked and glazed tiles can not be milled is angopl high strength . Terracotta tile waste, specific dimensions and surfaces are smooth .');
INSERT INTO `workshop_items2_full` VALUES ('314', 'Turkey', 'A', '38', 'Household waste, solid waste, waste oils from construction machines', 'Household waste, solid waste, waste oils from construction machines', 'EN', 'Material', '99 99 99', 'Have', null, null, null, 'Consulting made are from consulting ?firm .1000 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('315', 'Turkey', 'A', '38', '\nPackaging waste: Packaging waste collected by municipalities', '\nPackaging waste: Packaging waste collected by municipalities', 'EN', 'Material', '15 01 06', 'Have', null, null, null, 'Ak Oluklu Ambalaj: Wood pallets are re-evaluated . 2 tons / month');
INSERT INTO `workshop_items2_full` VALUES ('316', 'Turkey', 'A', '38', 'Solids', 'Solids', 'EN', 'Material', '99 99 99', 'Have', null, null, null, 'Do?a Plast: Stowage and shipment of materials can be produced for use in industrial companies.');
INSERT INTO `workshop_items2_full` VALUES ('317', 'Turkey', 'A', '38', 'Household waste', 'Household waste', 'EN', 'Material', '20 03 01', 'Have', null, null, null, 'ESO: Used in cement . Bien Ceramics (Bilecik ) above stands out. 5420 tons / year\nSeranit: Accumulated 500 tons and continuous exit.');
INSERT INTO `workshop_items2_full` VALUES ('318', 'Turkey', 'A', '38', '\nWaste batteries', '\nWaste batteries', 'EN', 'Material', '16 06 05', 'Have', null, null, null, 'Member companies of the waste resulting from the manufacture of food and textiles.');
INSERT INTO `workshop_items2_full` VALUES ('319', 'Turkey', 'A', '38', 'Electronic waste: Works should be done about collection of waste', 'Electronic waste: Works should be done about collection of waste', 'EN', 'Material', '16 02 14', 'Have', null, null, null, 'Deniz Döküm: Foundry sand , 20 tons / month . The hazardous nature of the content consists of silica sand + bentonite + coal dust.');
INSERT INTO `workshop_items2_full` VALUES ('320', 'Turkey', 'A', '38', '\nWaste oil: Oils and household waste oils in region', '\nWaste oil: Oils and household waste oils in region', 'EN', 'Material', '10 01 26', 'Have', null, null, null, 'ESO: Bilecik Demir Çelik Inc. slags resulting from the operator. Safe : 10:02:02. Induction slag. 1600 tons / month');
INSERT INTO `workshop_items2_full` VALUES ('321', 'Turkey', 'A', '38', 'Regional data', 'Regional data', 'EN', 'Unknown', '99 99 99', 'Have', null, null, null, 'Continuous waste from the ceramics factory . Inert , 17% humidity . 1000 tons / month');
INSERT INTO `workshop_items2_full` VALUES ('322', 'Turkey', 'A', '38', 'Consulting and investment support', 'Consulting and investment support', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Ford Otosan: Household waste ~ 30 tons / month . Used in the production of biogas and compost \nEski?ehir Büyük?ehir Belediyesi: Household waste ~ 800 tons / day.');
INSERT INTO `workshop_items2_full` VALUES ('323', 'Turkey', 'A', '38', '\nAsphalt and other road-building materials: To use for road construction of science works office', '\nAsphalt and other road-building materials: To use for road construction of science works office', 'EN', 'Material', '17 03 03', 'Want', null, null, null, 'ESO: Marble quarry is operating rubble waste (limestone) . It emerges from the Yüce Mining (Bilecik) . Cement may be the raw material . 500,000 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('324', 'Turkey', 'A', '38', 'Fertilizer waste', 'Fertilizer waste', 'EN', 'Material', '06 10 99', 'Want', null, null, null, 'Seranit: Analysis of waste to do.');
INSERT INTO `workshop_items2_full` VALUES ('325', 'Turkey', 'A', '38', 'Packaging waste', 'Packaging waste', 'EN', 'Material', '15 01 06', 'Want', null, null, null, 'Eski?ehir Endüstriyel Enerji: Safe G4 type filters .\nTepaba?? Belediyesi: 92,000 tons / year .');
INSERT INTO `workshop_items2_full` VALUES ('326', 'Turkey', 'A', '38', '\nWood, timber, fertilizer: Can be used in Municipalities and rural services directorate', '\nWood, timber, fertilizer: Can be used in Municipalities and rural services directorate', 'EN', 'Material', '15 01 03', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('327', 'Turkey', 'A', '38', 'Training', 'Training', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Tepeba?? Belediyesi: Hazardous waste and waste oils ( 70 tonnes / year).\nNuri?: Arising from the manufacture of automotive parts.');
INSERT INTO `workshop_items2_full` VALUES ('328', 'Turkey', 'A', '38', 'Service', 'Service', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Arçelik: Anhydrous, 1st category . To Koza sold at the moment. 55 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('329', 'Turkey', 'A', '38', 'Packing certification services', 'Packing certification services', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('330', 'Turkey', 'A', '38', 'Saplings and seeds', 'Saplings and seeds', 'EN', 'Material', '02 01 03', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('331', 'Turkey', 'A', '38', 'Consultancy: Can take consultancy services in carbon footprint', 'Consultancy: Can take consultancy services in carbon footprint', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Anadolu Üniversitesi: Present in the solid waste laboratories on GC- MS elemental analysis , calorimetry, oxygen permeability , extruders, including laboratory analysis of FCP- MS analysis support apparatus given in research projects.\n( A part of the Lab is accredited )\nTÜLOMSA?: Destructive / non destructive laboratory services . Mechanical - endete can be served on inspection. Accredited also to Destructive will be accredited in the future.\n?lksem Mühendislik: Jet mill , ball mill, high temperature ovens , the use of devices such as pH meter');
INSERT INTO `workshop_items2_full` VALUES ('332', 'Turkey', 'A', '39', 'Slasher powder and splint: Slasher powder and thin shavings waste from wood processed factory plants.', 'Slasher powder and splint: Slasher powder and thin shavings waste from wood processed factory plants.', 'EN', 'Material', '03 01 05', 'Have', '20000', 'tons', 'per year', 'Industrial Symbiosis and advice on waste management , project management and so on. the delivery of services');
INSERT INTO `workshop_items2_full` VALUES ('333', 'Turkey', 'A', '39', 'Waste marble: Abondoned mines. Non economic waste form mines of marble.', 'Waste marble: Abondoned mines. Non economic waste form mines of marble.', 'EN', 'Material', '01 01 01', 'Have', null, null, null, 'Chemical and physical analysis. Mineralogical analysis ( XRD and XRF ) service. Accredited ceramic final product retests. Boron analysis services provided.');
INSERT INTO `workshop_items2_full` VALUES ('334', 'Turkey', 'A', '39', 'Wood, trunk, fertilizer', 'Wood, trunk, fertilizer', 'EN', 'Material', '03 01 01', 'Have', null, null, null, 'Sabiha Gökçen MTAL: Across the public and private sector organizations in the preparation of national and international projects and cooperation in the implementation .\n?lksem Mühendislik: Project writing, quality management and executive training');
INSERT INTO `workshop_items2_full` VALUES ('335', 'Turkey', 'A', '39', 'Waste of production for wood products,waste of woods: 20.000 tonnes/year can provide broken pieces of stock', 'Waste of production for wood products,waste of woods: 20.000 tonnes/year can provide broken pieces of stock', 'EN', 'Material', '15 01 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('336', 'Turkey', 'A', '39', 'Advisory: Advisory for erosion controll and landscape,technics for lowering the carbon footprint. May useful for OZI companies.', 'Advisory: Advisory for erosion controll and landscape,technics for lowering the carbon footprint. May useful for OZI companies.', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Erosion control and landscaping issues and technical consulting services in use to reduce the carbon footprint. Utilization of companies OSB can be useful .');
INSERT INTO `workshop_items2_full` VALUES ('337', 'Turkey', 'A', '39', 'Young plant and seed.', 'Young plant and seed.', 'EN', 'Material', '02 01 03', 'Have', null, null, null, 'Submission services to public and private vocational training institutions. Computer , electronic and specialized technicians supply of aircraft airframe- engine subject .');
INSERT INTO `workshop_items2_full` VALUES ('338', 'Turkey', 'A', '39', 'Slasher powder: Can be given to forest peasant as combustable by pressing slasher powder with certain chemicals.', 'Slasher powder: Can be given to forest peasant as combustable by pressing slasher powder with certain chemicals.', 'EN', 'Material', '15 01 03', 'Want', '10000', 'tons', 'per year', 'Accredited educational services for welder training ( 17024 )');
INSERT INTO `workshop_items2_full` VALUES ('339', 'Turkey', 'A', '39', 'Soil for young plants.', 'Soil for young plants.', 'EN', 'Material', '17 05 04', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('340', 'Turkey', 'A', '39', 'Education', 'Education', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Ekolojik Enerji: Intermadiate storage can be.');
INSERT INTO `workshop_items2_full` VALUES ('341', 'Turkey', 'A', '39', 'Project Management and advisory services', 'Project Management and advisory services', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('342', 'Turkey', 'A', '40', 'Hazardous / non-hazardous waste', 'Hazardous / non-hazardous waste', 'EN', 'Material', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('343', 'Turkey', 'A', '40', 'Electrical / electronic materials: Can provide services especially on transportation of hazardous waste ', 'Electrical / electronic materials: Can provide services especially on transportation of hazardous waste ', 'EN', 'Material', '16 02 14', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('344', 'Turkey', 'A', '40', 'Waste transport services: Metal-plastic barrels, drums, IBC tank', 'Waste transport services: Metal-plastic barrels, drums, IBC tank', 'EN', 'Service', '19 12 12', 'Have', null, null, null, 'Ekobord Yeni Nesil Levha of plate ( Bilecik) are the resources that it can get . 60 tons / month can receive up to ..');
INSERT INTO `workshop_items2_full` VALUES ('345', 'Turkey', 'A', '40', 'Industrial packaging recycling service', 'Industrial packaging recycling service', 'EN', 'Service', '19 12 12', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('346', 'Turkey', 'A', '40', 'Waste management consultancy services', 'Waste management consultancy services', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'For use in the manufacture of feeder liners . 1 big bags / month .');
INSERT INTO `workshop_items2_full` VALUES ('347', 'Turkey', 'A', '40', '\nIndustrial packaging, metal / plastic barrels, drums IBC tank: To prepare for re-use', '\nIndustrial packaging, metal / plastic barrels, drums IBC tank: To prepare for re-use', 'EN', 'Tools', '15 01 05', 'Want', null, null, null, 'Receive services related to recycling technologies for the establishment of the center to provide a waste of waste that may be systematically collected from households and installation of the system.');
INSERT INTO `workshop_items2_full` VALUES ('348', 'Turkey', 'A', '40', 'Core sand', 'Core sand', 'EN', 'Material', '10 09 08', 'Want', null, null, null, 'Ford Otosan: OZI plant to be worked on .');
INSERT INTO `workshop_items2_full` VALUES ('349', 'Turkey', 'A', '40', 'Used cans', 'Used cans', 'EN', 'Tools', '15 01 06', 'Want', null, null, null, 'Aksoylu Trayler: Packaging waste and used wood pellets\nTepeba?? Belediyesi: Different characters on packaging waste (plastic , paper, etc. ). ~ 15,000 tons / year.');
INSERT INTO `workshop_items2_full` VALUES ('350', 'Turkey', 'A', '40', 'Waste and hazardous waste management', 'Waste and hazardous waste management', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Ekobord Yeni Nesil Levha of plate ( Bilecik) are the resources that it can get . 60 tons / month .');
INSERT INTO `workshop_items2_full` VALUES ('351', 'Turkey', 'A', '40', 'Machinery / Equipment', 'Machinery / Equipment', 'EN', 'Tools', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('352', 'Turkey', 'A', '41', '\nSludge: Currently it is given to ecological energy. 30-35% dryness', '\nSludge: Currently it is given to ecological energy. 30-35% dryness', 'EN', 'Material', '02 01 01', 'Have', null, null, null, 'Possibility to access EU funds for international partners and funds to develop the process. Industrial Symbiosis scope to connect with overseas companies.');
INSERT INTO `workshop_items2_full` VALUES ('353', 'Turkey', 'A', '41', 'Chemical contaminated drums / barrels: It is given to Özvar Endüstriyel At?k Ambalaj', 'Chemical contaminated drums / barrels: It is given to Özvar Endüstriyel At?k Ambalaj', 'EN', 'Tools', '15 01 10', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('354', 'Turkey', 'A', '41', 'Waste water: After pre-treatment, it is discharged to the organized industrial zone discharge point.', 'Waste water: After pre-treatment, it is discharged to the organized industrial zone discharge point.', 'EN', 'Material', '02 01 99', 'Have', null, null, null, 'K?vanç: Evaluated by the transport solution.\nTTGV: Bilecik Demir Çelik can evaluate');
INSERT INTO `workshop_items2_full` VALUES ('355', 'Turkey', 'A', '41', '\nConsultancy: Wastewater treatment plant and investments', '\nConsultancy: Wastewater treatment plant and investments', 'EN', 'Service', '02 01 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('356', 'Turkey', 'A', '41', 'Packing certification services', 'Packing certification services', 'EN', 'Service', '02 01 99', 'Want', null, null, null, '100 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('357', 'Turkey', 'A', '42', 'Paper', 'Paper', 'EN', 'Material', '20 01 01', 'Have', '2', 'tons', 'per year', 'While molding material used in the production of feeder liners in foundry.');
INSERT INTO `workshop_items2_full` VALUES ('358', 'Turkey', 'A', '42', 'plastic', 'plastic', 'EN', 'Material', '20 01 39', 'Have', '0', 'tons', 'per year', 'Anka Toprak Ürünleri: According to the results of the analysis can be evaluated .');
INSERT INTO `workshop_items2_full` VALUES ('359', 'Turkey', 'A', '42', 'metal waste', 'metal waste', 'EN', 'Material', '20 01 40', 'Have', '0', 'tons', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('360', 'Turkey', 'A', '42', 'Training service: Vocational training services for public and private sectors', 'Training service: Vocational training services for public and private sectors', 'EN', 'Service', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('361', 'Turkey', 'A', '42', 'Human Resources (Technical Staff): Specialized technicians on computer, electronic and aircraft engine issues', 'Human Resources (Technical Staff): Specialized technicians on computer, electronic and aircraft engine issues', 'EN', 'Service', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('362', 'Turkey', 'A', '42', 'Consulting and investment support', 'Consulting and investment support', 'EN', 'Service', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('363', 'Turkey', 'A', '42', 'Project and Consultancy Services: Project partnership with public and private organizations', 'Project and Consultancy Services: Project partnership with public and private organizations', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Metal waste associated with the manufacturing industry , consultancy on the use of the ceramic sector.\nConsultancy in the production of methane from animal waste');
INSERT INTO `workshop_items2_full` VALUES ('364', 'Turkey', 'A', '42', 'Training: ICT, aircraft maintenance, general education and vocational courses for teachers', 'Training: ICT, aircraft maintenance, general education and vocational courses for teachers', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('365', 'Turkey', 'A', '43', 'Household waste', 'Household waste', 'EN', 'Material', '20 03 01', 'Have', null, null, null, 'Low levels of R & D activities in order to assess');
INSERT INTO `workshop_items2_full` VALUES ('366', 'Turkey', 'A', '43', 'Food waste', 'Food waste', 'EN', 'Material', '02 03 04', 'Want', null, null, null, 'Iron, chromium, manganese content lower materials.\nSand , quartzite, limestone , marble, iron, aluminum, sodium and potassium content is high materials.');
INSERT INTO `workshop_items2_full` VALUES ('367', 'Turkey', 'A', '43', '\nRind: ESO: Dried pulp and peel waste could be qualified animal feed.', '\nRind: ESO: Dried pulp and peel waste could be qualified animal feed.', 'EN', 'Material', '02 03 04', 'Want', null, null, null, 'They are caused by the dye house waste . 45 tons / year .');
INSERT INTO `workshop_items2_full` VALUES ('368', 'Turkey', 'A', '44', '\nTesting / Analysis: Chemical and physical analysis. Mineralogical analysis (XRD and XRF) service. Accredited ceramic final product tests. Boron analysis services.', '\nTesting / Analysis: Chemical and physical analysis. Mineralogical analysis (XRD and XRF) service. Accredited ceramic final product tests. Boron analysis services.', 'EN', 'Service', '17 09 04', 'Have', null, null, null, 'ET?: Pulp production is due to the treatment of the process water sourced . 50% kuruluktad?r.15 m³ / day.\nEski?ehir OSB: 80% of dryness. There are 2000-2400 kcal thermal value. 30-40 tons / day.');
INSERT INTO `workshop_items2_full` VALUES ('369', 'Turkey', 'A', '44', 'Expertise / Consulting: Ensuring training coordination for energy efficiency in ceramics industry', 'Expertise / Consulting: Ensuring training coordination for energy efficiency in ceramics industry', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Waste from water-based paints are coming out of the printing plant.');
INSERT INTO `workshop_items2_full` VALUES ('370', 'Turkey', 'A', '45', 'Production waste: Accumulated waste, can be used as samples for R & D activities of alternative products made with ceramic materials ', 'Production waste: Accumulated waste, can be used as samples for R & D activities of alternative products made with ceramic materials ', 'EN', 'Material', '17 01 03', 'Have', null, null, null, 'Waste and hazardous waste management, AEEE management, decision support systems, sewage sludge, life cycle analysis etc. It can be carried out joint projects with companies on issues .');
INSERT INTO `workshop_items2_full` VALUES ('371', 'Turkey', 'A', '45', 'Filter press sludge: Continuous waste from the ceramics factory. Inert, 17% humidity.', 'Filter press sludge: Continuous waste from the ceramics factory. Inert, 17% humidity.', 'EN', 'Material', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('372', 'Turkey', 'A', '45', 'Plastic big bags, sacks: Seranit and ESÇEV can work together', 'Plastic big bags, sacks: Seranit and ESÇEV can work together', 'EN', 'Tools', '15 01 02', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('373', 'Turkey', 'A', '45', 'Hazardous recyclable waste: Already working with ESÇEV', 'Hazardous recyclable waste: Already working with ESÇEV', 'EN', 'Material', '15 01 06', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('374', 'Turkey', 'A', '45', 'Natural mineral waste: Already working with ESÇEV', 'Natural mineral waste: Already working with ESÇEV', 'EN', 'Material', '06 13 99', 'Have', null, null, null, '1 tonne / week');
INSERT INTO `workshop_items2_full` VALUES ('375', 'Turkey', 'A', '45', 'Asphalt and other road-building materials: Analysis will be done', 'Asphalt and other road-building materials: Analysis will be done', 'EN', 'Material', '17 03 02', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('376', 'Turkey', 'A', '45', '\nWaste oil', '\nWaste oil', 'EN', 'Material', '12 01 10', 'Want', '1750', 'tons', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('377', 'Turkey', 'A', '45', 'Consultancy', 'Consultancy', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('378', 'Turkey', 'A', '46', 'Electrical / electronic materials: Supports needed for setting-up the system of Municipality to collect the waste and starting the  recycling process', 'Electrical / electronic materials: Supports needed for setting-up the system of Municipality to collect the waste and starting the  recycling process', 'EN', 'Material', '16 02 16', 'Have', null, null, null, 'Kraft Ofset: 500-1000 kg / month;Ak Geri Dönü?üm: 30 tons / day');
INSERT INTO `workshop_items2_full` VALUES ('379', 'Turkey', 'A', '46', 'Waste batteries', 'Waste batteries', 'EN', 'Material', '16 06 02', 'Have', '2', 'tons', 'per year', 'Qualified-nonqualified');
INSERT INTO `workshop_items2_full` VALUES ('380', 'Turkey', 'A', '46', 'Solid waste', 'Solid waste', 'EN', 'Material', '20 03 01', 'Have', '92000', 'tons', 'per year', '800-1000 kg / year');
INSERT INTO `workshop_items2_full` VALUES ('381', 'Turkey', 'A', '46', 'Packaging waste', 'Packaging waste', 'EN', 'Material', '15 01 01', 'Have', '7500', 'tons', 'per year', 'Sawdust and wood processing workshops and chip waste from the factory are fine . 20,000 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('382', 'Turkey', 'A', '46', 'Waste paper (packaging waste)', 'Waste paper (packaging waste)', 'EN', 'Material', '15 01 01', 'Have', '700', 'tons', 'per day', null);
INSERT INTO `workshop_items2_full` VALUES ('383', 'Turkey', 'A', '46', 'Hazardous / non-hazardous waste', 'Hazardous / non-hazardous waste', 'EN', 'Material', '99 99 99', 'Have', null, null, null, '7500 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('384', 'Turkey', 'A', '46', '\nHazardous recyclable waste', '\nHazardous recyclable waste', 'EN', 'Material', '99 99 99', 'Have', null, null, null, 'Orman Bölge Müdürlü?ü: 20,000 tons / year can provide up broken branches .\nAk Geri Dönü?üm: Wood waste can provide .');
INSERT INTO `workshop_items2_full` VALUES ('385', 'Turkey', 'A', '46', 'Electronic waste', 'Electronic waste', 'EN', 'Material', '16 02 14', 'Have', null, null, null, 'Can take all .');
INSERT INTO `workshop_items2_full` VALUES ('386', 'Turkey', 'A', '46', '\nWaste vegetable oil', '\nWaste vegetable oil', 'EN', 'Material', '20 01 25', 'Have', '70', 'tons', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('387', 'Turkey', 'A', '46', 'Waste oil: Hazardous waste', 'Waste oil: Hazardous waste', 'EN', 'Material', '13 01 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('388', 'Turkey', 'A', '46', 'Training', 'Training', 'EN', 'Service', '99 99 99', 'Have', null, null, null, 'Small metal particles can be analyzed received .');
INSERT INTO `workshop_items2_full` VALUES ('389', 'Turkey', 'A', '46', 'Packaging waste: Support for systematically collecting the packing wastes from houses, setting-up a collection methodology and bringing it into the recycling process', 'Packaging waste: Support for systematically collecting the packing wastes from houses, setting-up a collection methodology and bringing it into the recycling process', 'EN', 'Service', '15 01 06', 'Want', null, null, null, 'Ak Geri Dönü?üm: can take it all');
INSERT INTO `workshop_items2_full` VALUES ('390', 'Turkey', 'A', '46', 'Integrated waste (packaging-waste oil and electronic waste etc.): Getting services for establishing a waste collection center of waste from households and installing the system', 'Integrated waste (packaging-waste oil and electronic waste etc.): Getting services for establishing a waste collection center of waste from households and installing the system', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('391', 'Turkey', 'A', '46', 'Grinding and sanding dust: Can discuss for road construction', 'Grinding and sanding dust: Can discuss for road construction', 'EN', 'Material', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('392', 'Turkey', 'A', '46', 'Grease trap waste: Municipality can advise on this subject', 'Grease trap waste: Municipality can advise on this subject', 'EN', 'Material', '16 10 04', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('393', 'Turkey', 'A', '46', 'Training service', 'Training service', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('394', 'Turkey', 'A', '46', 'Saplings and seeds', 'Saplings and seeds', 'EN', 'Material', '02 01 03', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('395', 'Turkey', 'A', '46', '\nWaste and hazardous waste management', '\nWaste and hazardous waste management', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('396', 'Turkey', 'A', '46', 'Consultancy: Can take consultancy on mine rehabilitation', 'Consultancy: Can take consultancy on mine rehabilitation', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'By compressed sawdust and various chemicals may be pressed by the forest villagers as firewood . 10,000 tons / year');
INSERT INTO `workshop_items2_full` VALUES ('397', 'Turkey', 'A', '47', 'Slag waste: Iron and steel slags from Bilecik. Non-hazardous. 10:02:02. Induction slag.', 'Slag waste: Iron and steel slags from Bilecik. Non-hazardous. 10:02:02. Induction slag.', 'EN', 'Material', '10 02 02', 'Have', '1600', 'tons', 'per month', null);
INSERT INTO `workshop_items2_full` VALUES ('398', 'Turkey', 'A', '47', 'Ceramic fractures: Can be used for cement. From Bien Ceramics Bilecik Plant', 'Ceramic fractures: Can be used for cement. From Bien Ceramics Bilecik Plant', 'EN', 'Material', '10 02 02', 'Have', '5420', 'tons', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('399', 'Turkey', 'A', '47', 'Debris waste: Rubble waste (limestone) from marble quarry plant (Yuce Maden Co.). Can be used as cement raw material', 'Debris waste: Rubble waste (limestone) from marble quarry plant (Yuce Maden Co.). Can be used as cement raw material', 'EN', 'Material', '01 01 02', 'Have', '500000', 'tons', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('400', 'Turkey', 'A', '47', 'Cellulose, cardboard, kraft cellulose such as cement bags: Ekobord Yeni Nesil Levha (Bilecik) can take this resource', 'Cellulose, cardboard, kraft cellulose such as cement bags: Ekobord Yeni Nesil Levha (Bilecik) can take this resource', 'EN', 'Material', '15 01 01', 'Want', '60', 'tons', 'per month', null);
INSERT INTO `workshop_items2_full` VALUES ('401', 'Turkey', 'A', '47', 'Regional data', 'Regional data', 'EN', 'Unknown', '99 99 99', 'Want', null, null, null, 'TÜLOMSA?: Wood packaging waste');
INSERT INTO `workshop_items2_full` VALUES ('402', 'Turkey', 'A', '47', 'Expertise-laboratory services', 'Expertise-laboratory services', 'EN', 'Service', '15 01 01', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('403', 'Turkey', 'A', '48', 'Wood packaging waste (Qualified / Non-qualified)', 'Wood packaging waste (Qualified / Non-qualified)', 'EN', 'Material', '15 01 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('404', 'Turkey', 'A', '48', 'Hazardous / Non-hazardous metal waste, wood chips', 'Hazardous / Non-hazardous metal waste, wood chips', 'EN', 'Material', '17 02 01', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('405', 'Turkey', 'A', '48', 'Waste copper, cables', 'Waste copper, cables', 'EN', 'Material', '12 01 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('406', 'Turkey', 'A', '48', 'Reusable packaging waste', 'Reusable packaging waste', 'EN', 'Material', '15 01 05', 'Have', null, null, null, 'Metal shavings polystyrene foam stands out as sawdust. 1 big bags / month');
INSERT INTO `workshop_items2_full` VALUES ('407', 'Turkey', 'A', '48', 'Descaling', 'Descaling', 'EN', 'Unknown', '99 99 99', 'Have', null, null, null, 'H?zlan Makine: Aluminum shavings; Eski?ehir Endüstriyel Enerji: Non-hazardous metal waste thrown the case out of steam and gas turbine systems used; Arçelik: Sheet metal wastes ( iron, copper , aluminum ). Going to the landfill.');
INSERT INTO `workshop_items2_full` VALUES ('408', 'Turkey', 'A', '48', 'Hazardous / non-hazardous waste', 'Hazardous / non-hazardous waste', 'EN', 'Material', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('409', 'Turkey', 'A', '48', 'Various types of batteries', 'Various types of batteries', 'EN', 'Material', '16 06 02', 'Have', null, null, null, 'We supply the means of production to companies engaged in the production of biomass .');
INSERT INTO `workshop_items2_full` VALUES ('410', 'Turkey', 'A', '48', 'Waste oil', 'Waste oil', 'EN', 'Material', '13 02 05', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('411', 'Turkey', 'A', '48', 'Training of welders for welding manufacturing: Accredited educational services for welder training (17024)', 'Training of welders for welding manufacturing: Accredited educational services for welder training (17024)', 'EN', 'Service', '12 01 03', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('412', 'Turkey', 'A', '48', 'Marketing and sales consulting', 'Marketing and sales consulting', 'EN', 'Service', '99 99 99', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('413', 'Turkey', 'A', '48', 'Destructive / Nondestructive Laboratory Services: Providing services on mechanical-endete and destructive testing fields ', 'Destructive / Nondestructive Laboratory Services: Providing services on mechanical-endete and destructive testing fields ', 'EN', 'Service', '16 01 10', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('414', 'Turkey', 'A', '48', 'Sawmill dust and shavings', 'Sawmill dust and shavings', 'EN', 'Material', '15 01 01', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('415', 'Turkey', 'A', '48', 'Training and Consulting: Production and management processes', 'Training and Consulting: Production and management processes', 'EN', 'Service', '99 99 99', 'Want', null, null, null, 'Anka Toprak Ürünleri: It may take if given in dry');
INSERT INTO `workshop_items2_full` VALUES ('416', 'Turkey', 'A', '48', 'Environmental measures', 'Environmental measures', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('417', 'Turkey', 'A', '48', 'Industrial packaging recycling service', 'Industrial packaging recycling service', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('418', 'Turkey', 'A', '48', 'Waste transport services', 'Waste transport services', 'EN', 'Service', '99 99 99', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('419', 'Turkey', 'A', '49', 'Rind: Fruit peel, fruit syrups', 'Rind: Fruit peel, fruit syrups', 'EN', 'Material', '02 03 04', 'Have', '1', 'tons', 'per week', null);
INSERT INTO `workshop_items2_full` VALUES ('420', 'Turkey', 'A', '49', 'Palets, syrup tanks: Plastic, pallets suitable for food and syrup tank', 'Palets, syrup tanks: Plastic, pallets suitable for food and syrup tank', 'EN', 'Tools', '15 01 03', 'Want', null, null, null, null);
INSERT INTO `workshop_items2_full` VALUES ('422', 'Spain', 'B', '51', 'R I tiesto cocido ', 'R I tiesto cocido ', 'SP', 'Material', null, 'Have', '622', 'Tm', null, null);
INSERT INTO `workshop_items2_full` VALUES ('423', 'Spain', 'B', '51', 'Lodos cerámicos', 'Lodos cerámicos', 'SP', 'Material', null, 'Have', '6632', 'Tm', null, null);
INSERT INTO `workshop_items2_full` VALUES ('424', 'Spain', 'B', '51', 'Crudo - rechazo', 'Crudo - rechazo', 'SP', 'Material', null, 'Have', '2476', 'Tm', null, null);
INSERT INTO `workshop_items2_full` VALUES ('425', 'Spain', 'B', '51', 'R I papel-carton', 'R I papel-carton', 'SP', 'Material', null, 'Have', '16250', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('426', 'Spain', 'B', '51', 'R I plastico', 'R I plastico', 'SP', 'Material', null, 'Have', '11845', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('427', 'Spain', 'B', '51', 'R I maderas', 'R I maderas', 'SP', 'Material', null, 'Have', '12030', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('428', 'Spain', 'B', '51', 'R I chatarras', 'R I chatarras', 'SP', 'Material', null, 'Have', '6000', 'Tm', null, null);
INSERT INTO `workshop_items2_full` VALUES ('429', 'Spain', 'B', '51', 'RP Aceites', 'RP Aceites', 'SP', 'Material', null, 'Have', '700', 'Litros', null, null);
INSERT INTO `workshop_items2_full` VALUES ('430', 'Spain', 'B', '51', 'RP Aerosoles', 'RP Aerosoles', 'SP', 'Material', null, 'Have', '140', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('431', 'Spain', 'B', '51', 'RP aserrín contam.', 'RP aserrín contam.', 'SP', 'Material', null, 'Have', '3746', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('432', 'Spain', 'B', '51', 'RP Baterias usadas ', 'RP Baterias usadas ', 'SP', 'Material', null, 'Have', '66', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('433', 'Spain', 'B', '51', 'RP Disolventes', 'RP Disolventes', 'SP', 'Material', null, 'Have', '220', 'Litros', null, null);
INSERT INTO `workshop_items2_full` VALUES ('434', 'Spain', 'B', '51', 'RP Elem. Mercur.', 'RP Elem. Mercur.', 'SP', 'Material', null, 'Have', '29', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('435', 'Spain', 'B', '51', 'RP Env. metalicos ', 'RP Env. metalicos ', 'SP', 'Material', null, 'Have', '296', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('436', 'Spain', 'B', '51', 'RP plastico ', 'RP plastico ', 'SP', 'Material', null, 'Have', '2494', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('437', 'Spain', 'B', '51', 'RP papel', 'RP papel', 'SP', 'Material', null, 'Have', '1426', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('438', 'Spain', 'B', '51', 'RP Filtros aceite ', 'RP Filtros aceite ', 'SP', 'Material', null, 'Have', '62', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('439', 'Spain', 'B', '51', 'RP Prod. Cer. Caduc.', 'RP Prod. Cer. Caduc.', 'SP', 'Material', null, 'Have', '50748', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('440', 'Spain', 'B', '51', 'RP Tierra contam.', 'RP Tierra contam.', 'SP', 'Material', null, 'Have', '2268', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('441', 'Spain', 'B', '51', 'RP Trapos y absor.', 'RP Trapos y absor.', 'SP', 'Material', null, 'Have', '1596', 'Kg', null, null);
INSERT INTO `workshop_items2_full` VALUES ('442', 'Spain', 'B', '52', 'Used oil ', 'Used oil ', 'EN', 'Material', '12 01 10', 'Have', '255', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('443', 'Spain', 'B', '52', 'Cleaning solvents', 'Cleaning solvents', 'EN', 'Material', '99 99 99', 'Have', '150', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('444', 'Spain', 'B', '52', 'Greases s', 'Greases s', 'EN', 'Material', '99 99 99', 'Have', '3', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('445', 'Spain', 'B', '52', 'Inorganic absorbents s', 'Inorganic absorbents s', 'EN', 'Material', '99 99 99', 'Have', '38', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('446', 'Spain', 'B', '52', 'Contaminated rugs and cotton s', 'Contaminated rugs and cotton s', 'EN', 'Material', '04 02 22', 'Have', '382', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('447', 'Spain', 'B', '52', 'Empty aerosolss', 'Empty aerosolss', 'EN', 'Material', '15 01 04', 'Have', '24', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('448', 'Spain', 'B', '52', 'Plastic containerss', 'Plastic containerss', 'EN', 'Material', '15 01 02', 'Have', '28', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('449', 'Spain', 'B', '52', 'Oil filters s', 'Oil filters s', 'EN', 'Material', '16 01 07', 'Have', '78', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('450', 'Spain', 'B', '52', 'Contaminated air filterss', 'Contaminated air filterss', 'EN', 'Material', '15 02 02', 'Have', '46', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('451', 'Spain', 'B', '52', 'Contaminated raffia ', 'Contaminated raffia ', 'SP', 'Material', null, 'Have', '1', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('452', 'Spain', 'B', '52', 'Fluorescent tubes', 'Fluorescent tubes', 'EN', 'Material', '20 01 21', 'Have', '20', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('453', 'Spain', 'B', '52', 'Scrap', 'Scrap', 'EN', 'Material', '99 99 99', 'Have', '4', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('454', 'Spain', 'B', '52', 'Copper', 'Copper', 'EN', 'Material', '10 06 99', 'Have', '220', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('455', 'Spain', 'B', '52', 'Paper ', 'Paper ', 'EN', 'Material', '20 01 01', 'Have', '3', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('456', 'Spain', 'B', '52', 'Paper to destroy (documents) ', 'Paper to destroy (documents) ', 'EN', 'Material', '20 01 01', 'Have', '3', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('457', 'Spain', 'B', '52', 'Plastic (big-bags)', 'Plastic (big-bags)', 'EN', 'Tools', '15 01 02', 'Have', '39', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('458', 'Spain', 'B', '52', 'Inert waste', 'Inert waste', 'EN', 'Material', '10 13 14', 'Have', '184', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('459', 'Spain', 'B', '52', 'Organic waste', 'Organic waste', 'EN', 'Material', '08 01 12', 'Have', '5', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('460', 'Spain', 'B', '52', 'Bag filters', 'Bag filters', 'EN', 'Material', '15 01 02', 'Have', '104', 'Kg', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('461', 'Spain', 'B', '53', 'Paint and varnish containing organic solvents or other dangerous substances', 'Paint and varnish containing organic solvents or other dangerous substances', 'EN', 'Material', '08 01 11', 'Have', '0', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('462', 'Spain', 'B', '53', 'Absorbents, filter materials', 'Absorbents, filter materials', 'EN', 'Material', '15 02 02', 'Have', '1', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('463', 'Spain', 'B', '53', 'Glass waste containing heavy metals', 'Glass waste containing heavy metals', 'EN', 'Material', '10 12 11', 'Have', '215', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('464', 'Spain', 'B', '53', 'metal packaging', 'metal packaging', 'EN', 'Material', '15 01 11', 'Have', '1', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('465', 'Spain', 'B', '53', 'Packaging containing residues of substances hazardous or contaminated by them', 'Packaging containing residues of substances hazardous or contaminated by them', 'EN', 'Material', '15 01 10', 'Have', '12', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('466', 'Spain', 'B', '53', 'oil filters', 'oil filters', 'EN', 'Material', '16 01 07', 'Have', '0', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('467', 'Spain', 'B', '53', 'Insulation materials', 'Insulation materials', 'EN', 'Material', '17 06 03', 'Have', '0', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('468', 'Spain', 'B', '53', 'Engine oils', 'Engine oils', 'EN', 'Material', '13 02 08', 'Have', '0', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('469', 'Spain', 'B', '53', 'Discarded equipment', 'Discarded equipment', 'EN', 'Tools', '16 02 13', 'Have', '0', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('470', 'Spain', 'B', '53', 'inorganic wastes containing dangerous substances', 'inorganic wastes containing dangerous substances', 'EN', 'Material', '16 03 03', 'Have', '0', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('471', 'Spain', 'B', '53', 'Led batteries', 'Led batteries', 'EN', 'Material', '16 06 01', 'Have', '0', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('472', 'Spain', 'B', '53', 'Packaging paper and cardboard', 'Packaging paper and cardboard', 'EN', 'Material', '15 01 01', 'Have', '6', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('473', 'Spain', 'B', '53', 'Plastic bottles', 'Plastic bottles', 'EN', 'Material', '15 01 02', 'Have', '7', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('474', 'Spain', 'B', '53', 'Wooden containers', 'Wooden containers', 'EN', 'Tools', '15 01 03', 'Have', '67', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('475', 'Spain', 'B', '53', 'Waste ceramics, bricks, tiles and construction', 'Waste ceramics, bricks, tiles and construction', 'EN', 'Material', '10 12 08', 'Have', '24', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('476', 'Spain', 'B', '53', 'Iron and Steel', 'Iron and Steel', 'EN', 'Material', '17 04 05', 'Have', '3', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('477', 'Spain', 'B', '53', 'Water sludges containing ceramic materials', 'Water sludges containing ceramic materials', 'EN', 'Material', '08 02 02', 'Have', '476', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('478', 'Spain', 'B', '53', 'Water suspensions containing ceramic materials', 'Water suspensions containing ceramic materials', 'EN', 'Material', '08 02 03', 'Have', '77', 'tons', 'Year', null);
INSERT INTO `workshop_items2_full` VALUES ('479', 'Spain', 'B', '54', 'Phosphatising sludges', 'Phosphatising sludges', 'EN', 'Material', '11 01 08', 'Have', '6', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('480', 'Spain', 'B', '54', 'Sludges from waste water treatment plant', 'Sludges from waste water treatment plant', 'EN', 'Material', '11 01 09', 'Have', '8', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('481', 'Spain', 'B', '54', 'Machining sludges', 'Machining sludges', 'EN', 'Material', '12 01 14', 'Have', '4', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('482', 'Spain', 'B', '54', 'Used Oil', 'Used Oil', 'EN', 'Material', '13 02 05', 'Have', '34', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('483', 'Spain', 'B', '54', ' Dirty solvent', ' Dirty solvent', 'EN', 'Material', '14 06 03', 'Have', '2', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('484', 'Spain', 'B', '54', 'Contaminated plastic, metallic and paper packaging', 'Contaminated plastic, metallic and paper packaging', 'EN', 'Material', '15 01 10', 'Have', '1', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('485', 'Spain', 'B', '54', 'Empty aerosols', 'Empty aerosols', 'EN', 'Material', '15 01 11', 'Have', '0', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('486', 'Spain', 'B', '54', 'Contaminated cloth and used filters', 'Contaminated cloth and used filters', 'EN', 'Material', '15 02 02', 'Have', '2', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('487', 'Spain', 'B', '54', 'Used sepiolite', 'Used sepiolite', 'EN', 'Material', '15 02 02', 'Have', '3', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('488', 'Spain', 'B', '54', ' Expired products', ' Expired products', 'EN', 'Material', '16 05 06', 'Have', '0', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('489', 'Spain', 'B', '54', 'Quenching dross', 'Quenching dross', 'EN', 'Material', '11 03 02', 'Have', '12', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('490', 'Spain', 'B', '54', 'Wastes containing oil', 'Wastes containing oil', 'EN', 'Material', '16 07 08', 'Have', '2', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('491', 'Spain', 'B', '54', ' Fluorescent tubes and other mercurycontaining lighting', ' Fluorescent tubes and other mercurycontaining lighting', 'EN', 'Material', '20 01 21', 'Have', '0', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('492', 'Spain', 'B', '54', 'Electrical and electronic equipment', 'Electrical and electronic equipment', 'EN', 'Material', '20 01 35', 'Have', '0', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('493', 'Spain', 'B', '54', ' Decantation pool sludges', ' Decantation pool sludges', 'EN', 'Material', '19 02 05', 'Have', '14', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('494', 'Spain', 'B', '54', 'Scrap', 'Scrap', 'EN', 'Material', '17 04 07', 'Have', '365', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('495', 'Spain', 'B', '54', 'Powder coating waste', 'Powder coating waste', 'EN', 'Material', '08 01 12', 'Have', '5', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('496', 'Spain', 'B', '54', 'Paper and cardboard', 'Paper and cardboard', 'EN', 'Material', '15 01 01', 'Have', '25', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('497', 'Spain', 'B', '54', 'Shot peening dust and Steel dust', 'Shot peening dust and Steel dust', 'EN', 'Material', '12 01 01', 'Have', '271', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('498', 'Spain', 'B', '54', 'Wood', 'Wood', 'EN', 'Material', '15 01 03', 'Have', '24', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('499', 'Spain', 'B', '54', 'Plastic', 'Plastic', 'EN', 'Material', '15 01 02', 'Have', '8', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('500', 'Spain', 'B', '54', 'General waste', 'General waste', 'EN', 'Material', '19 02 03', 'Have', '25', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('501', 'Spain', 'B', '54', ' Non-hazardous electrical and electronic equipment', ' Non-hazardous electrical and electronic equipment', 'EN', 'Material', '20 03 07', 'Have', '0', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('502', 'Spain', 'B', '54', 'Copper', 'Copper', 'EN', 'Material', '17 04 01', 'Have', '0', 'tons ', 'Month', null);
INSERT INTO `workshop_items2_full` VALUES ('504', 'Spain', 'C', '55', 'Water', 'Agua: Agua', 'SP', 'Material', null, 'Have', '10', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('505', 'Spain', 'C', '55', 'Experience in R&D management', 'Experiencia en gestión de I+D+i', 'SP', 'Service', '06 13 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('506', 'Spain', 'C', '55', 'Energie Hot steam T=250 ºC', 'Energía Focos calientes: T=250 ºC', 'SP', 'Energy', '06 13 99', 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('507', 'Spain', 'C', '55', 'Cardboard container', 'Envase Cartón', 'SP', 'Material', '06 13 99', 'Have', '13', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('508', 'Spain', 'C', '55', 'Waste wood pallet', 'Madera: Residuo de pallets', 'SP', 'Tools', '06 13 99', 'Have', '17', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('509', 'Spain', 'C', '55', 'Ceramic flower pot ', 'Cerámica Tiesto cocido', 'SP', 'Material', '06 13 99', 'Have', '491', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('510', 'Spain', 'C', '55', 'Water sludge Not known solid content', 'Agua Lodos: Agua residual: lodos\nNo se sabe el contenido en sólidos', 'SP', 'Material', '06 13 99', 'Have', '5781', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('511', 'Spain', 'C', '55', 'Metal scrap', 'Metal Chatarra', 'SP', 'Material', '06 13 99', 'Have', '7', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('512', 'Spain', 'C', '55', 'Plastic', 'Plástico', 'SP', 'Material', '06 13 99', 'Have', '8', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('513', 'Spain', 'C', '55', 'Logistical waste management: joint management of wood plastics etc.', 'Logística Gestión Residuos: Gestión conjunta (maderas, plásticos, etc.)', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('514', 'Spain', 'C', '55', 'Land asphalted land', 'Terreno: Terreno asfaltado', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('515', 'Spain', 'C', '56', 'Grinding capacity: milling and micronization of materials', 'Capacidad Molienda: Molienda y micronización de materiales', 'SP', 'Service', null, 'Have', '10000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('516', 'Spain', 'C', '56', 'Mineral inert residue; inert residue grinding balls', 'Mineral Residuo inerte: Residuo inerte bolas de molienda', 'SP', 'Material', '06 13 99', 'Have', '177', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('517', 'Spain', 'C', '56', 'Container loading dock in the port of Castellón', 'Terreno Muelle de carga de contenedores: En el puerto de Castellón', 'SP', 'Service', '06 13 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('518', 'Spain', 'C', '56', 'Land in the port of Castellón', 'Terreno Puerto: En el puerto de Castellón', 'SP', 'Service', '06 13 99', 'Have', '14500', 'Square Metres (m2)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('519', 'Spain', 'C', '56', 'Packaging Plastic waste', 'Plástico Residuo: De embalaje…', 'SP', 'Material', '06 13 99', 'Have', '37', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('520', 'Spain', 'C', '56', 'Land in Nules; Polígono La Mina', 'Terreno Nules: Polígono La Mina', 'SP', 'Service', '06 13 99', 'Have', '22000', 'Square Metres (m2)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('521', 'Spain', 'C', '56', 'Wooden pallets; pallets in good condition but colored', 'Madera Pallets: Pallets en buen estado pero coloreados', 'SP', 'Tools', '03 01 99', 'Want', '1200', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('522', 'Spain', 'C', '56', 'Silo storage capacity ', 'Capacidad Almacenamiento: Silos', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('523', 'Spain', 'C', '56', 'Experience in ceramic coatings', 'Experiencia Materiales: Experiencia en recubrimientos cerámicos', 'SP', 'Service', null, 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('524', 'Spain', 'C', '57', 'Frit waste: Remains of ceramic frits', 'Frita Residuo: Restos de fritas cerámicas', 'SP', 'Material', '06 13 99', 'Have', '200', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('525', 'Spain', 'C', '57', 'Ceramic water: ceramaic purified water', 'Agua cerámica: Aguas de depuración cerámica', 'SP', 'Material', '19 08 99', 'Have', '500', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('526', 'Spain', 'C', '57', 'Wood packaging', 'Madera envases: Envases de madera', 'SP', 'Material', '03 01 01', 'Have', '67', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('527', 'Spain', 'C', '57', 'Energy from steam heat: waste heat at temperature below 200ºC', 'Energía Calor Residual: Calor residual a temperatura inferior a 200ºC', 'SP', 'Energy', null, 'Have', '99', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('528', 'Spain', 'C', '57', 'ZZZ gas CO2', 'ZZZ Gases CO2: Gases de CO2', 'SP', 'Material', null, 'Have', '12', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('529', 'Spain', 'C', '57', 'Waste Plastic', 'Plástico: Residuos de plástico', 'SP', 'Material', '15 01 02', 'Have', '6', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('530', 'Spain', 'C', '57', 'Transport logistics', 'Logística Transporte', 'SP', 'Service', null, 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('531', 'Spain', 'C', '57', 'FRX analysis capacity: FRX analytical laboratory equipment', 'Capacidad Análisis FRX: Equipos de laboratorio de análisis FRX', 'SP', 'Service', null, 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('532', 'Spain', 'C', '57', 'Experience in Standards and Certifications: Quality, Environment, ISO regulations', 'Experiencia Normativa: Experiencia en Calidad, Medio Ambiente, Normativa ISO', 'SP', 'Service', null, 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('533', 'Spain', 'C', '57', 'Energy Gas: Natural gas at low cost', 'Energía Gas: Gas Natural a bajo coste', 'SP', 'Energy', null, 'Want', '500', 'MW', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('534', 'Spain', 'C', '57', 'Logistical waste transport: share transfer in waste management 1 truck per month', 'Logística Transporte Residuos: Compartir traslado en gestión de residuos. 1 camion al mes', 'SP', 'Service', null, 'Want', '12', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('535', 'Spain', 'C', '57', 'Compressed air capacity', 'Capacidad Aire Comprimido: Aire Comprimido', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('536', 'Spain', 'C', '57', 'Energy Electricity: low cost electricity', 'Energía Electricidad: Electricidad a bajo coste', 'SP', 'Energy', null, 'Want', '5000', 'kWh', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('537', 'Spain', 'C', '58', 'Mineral Glass Raw material mixed', 'Mineral Materia prima de vidrio: Mezclada', 'SP', 'Material', '10 11 03', 'Have', '500', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('538', 'Spain', 'C', '58', 'Inorganic Cobalt Sulphate', 'Inorgánico Sulfato de cobalto', 'SP', 'Material', '06 13 99', 'Have', '1', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('539', 'Spain', 'C', '58', 'Inorganic Chromium Oxide', 'Inorgánico Óxido de cromo', 'SP', 'Material', '06 13 99', 'Have', '1', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('540', 'Spain', 'C', '58', 'Glass Powder: Powder (medium size 1 mm)', 'Vidrio En polvo: En polvo (tamaño medio 1 mm)', 'SP', 'Material', '06 13 99', 'Have', '100', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('541', 'Spain', 'C', '58', 'Inorganic Selenium metal: In pellets', 'Inorgánico Selenio metálico: En pellets', 'SP', 'Material', '10 11 99', 'Have', '1', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('542', 'Spain', 'C', '58', 'Chopped Mirror Glass', 'Vidrio De Espejo troceado', 'SP', 'Material', '10 11 12', 'Have', '600', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('543', 'Spain', 'C', '58', 'Cardboard: In slats', 'Cartón: En listones', 'SP', 'Material', '10 11 99', 'Have', '8', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('544', 'Spain', 'C', '58', 'Experience Treatment Discharge: Experience in treatment of discharges neutralization and hydrofluoric acid', 'Experiencia Tratamiento Vertidos: Experiencia en tratamiento de vertidos neutralización y ácido fluorhídrico', 'SP', 'Service', null, 'Want', '2', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('545', 'Spain', 'C', '58', 'Experience Granulometric Analysis: Experience in large-scale granulometric analysis. 100 kg per sample', 'Experiencia Análisis Granulométrico: Experiencia en Análisis Granulométrico a gran escala. 100 Kg por muestra', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('546', 'Spain', 'C', '58', 'Experience Quality: Quality of painted surfaces', 'Experiencia Calidad: Calidad de superficies pintadas', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('547', 'Spain', 'C', '59', 'Experience Advice: Legal environmental consultancy', 'Experiencia Asesoramiento: Asesoramiento Legal en Medio Ambiente', 'SP', 'Service', null, 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('548', 'Spain', 'C', '59', 'Capacity Ion retardation: Semi-industrial ion retardation equipment. It is used to regenerate acids contaminated with metals', 'Capacidad Retardo iónico: Equipo de retardo iónico semi-industrial. Se utiliza para regenerar ácidos contaminados con metales', 'SP', 'Service', null, 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('549', 'Spain', 'C', '59', 'Experience Safety: Product safety advice (CE marking)', 'Experiencia Seguridad: Asesoramiento seguridad de productos (marcado CE)', 'SP', 'Service', null, 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('550', 'Spain', 'C', '59', 'Standards and certification services: Experience in ISO14001, ISO9001, Energy', 'Experiencia Normativa: Experiencia en ISO14001, ISO9001, Energía', 'SP', 'Service', null, 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('551', 'Spain', 'C', '59', 'Prototyping Capacity: Manufacturing of product components of Metal and plastic ', 'Capacidad Prototipado: Servicios de Fabricación Aditiva de metal y plástico', 'SP', 'Service', null, 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('552', 'Spain', 'C', '59', 'Vacuum Evaporation Capacity: Semi-industrial Vacuum Evaporation Equipment', 'Capacidad Evaporación Vacío: Equipo de Evaporación al vacío semiindustrial', 'SP', 'Service', null, 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('553', 'Spain', 'C', '59', ' Metal Scrap', 'Metales Chatarra: Chatarra metálica', 'SP', 'Material', '06 04 99', 'Have', '500', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('554', 'Spain', 'C', '59', 'Capacity Students: Students for master of Management and Treatment of industrial water', 'Capacidad Alumnos: Alumnos para master de Gestión y Tratamiento de aguas industriales', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('555', 'Spain', 'C', '59', 'Capacity Valorisation metals: Valorisation of metal scrap', 'Capacidad Valorización metales: Valorización de chatarra metálica', 'SP', 'Service', null, 'Want', '0', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('556', 'Spain', 'C', '59', 'Capacity Internships: Companies to do master\'s internship', 'Capacidad Prácticas: Empresas para hacer prácticas de máster', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('557', 'Spain', 'C', '59', 'Capacity Visits: Companies to make visits', 'Capacidad Visitas: Empresas para hacer visitas', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('558', 'Spain', 'C', '59', 'Capacity Partners projects: Technological partners for R & D projects', 'Capacidad Socios proyectos: Socios Tecnológicos para proyectos de I+D', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('559', 'Spain', 'C', '59', 'Capacity Projects: Companies to participate in activities and projects of Industrial Symbiosis, TRIS.', 'Capacidad Proyectos: Empresas para participar en actividades y proyetos de Simbiosis Industrial, TRIS.', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('560', 'Spain', 'C', '60', 'Inorganic construction waste', 'Inorgánicos Residuos Construcción: Materiales residuos de construcción', 'SP', 'Material', '06 13 99', 'Have', '5000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('561', 'Spain', 'C', '60', 'Transport Capacity: logistics', 'Logística Transporte: Logística', 'SP', 'Service', null, 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('562', 'Spain', 'C', '60', 'Capacity Crushing: Crushing of Ceramic Materials', 'Capacidad Trituración: Trituración de Materiales Cerámicos', 'SP', 'Service', null, 'Have', '10000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('563', 'Spain', 'C', '60', 'Grinding Ceramics: Porcelain Grinding and Scrubbing', 'Cerámica Abrasión: Fregaduras de Abrasión Porcelánica', 'SP', 'Material', '06 13 99', 'Have', '1000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('564', 'Spain', 'C', '60', 'Ceramics Waste: Ceramic, tile, brick waste', 'Cerámica Residuos: Residuos de cerámica, tejas, ladrillos', 'SP', 'Material', null, 'Want', '1000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('565', 'Spain', 'C', '60', 'Land Storage: Land for storing materials', 'Terreno Almacenamiento: Terreno para acopio de materiales', 'SP', 'Service', null, 'Want', '3000', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('566', 'Spain', 'C', '60', 'Minerals Limestone: Limestone', 'Minerales Piedra Caliza: Piedra Caliza', 'SP', 'Material', null, 'Want', '2000', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('567', 'Spain', 'C', '60', 'Grinding Capacity: Lower Granulometry in ceramic waste', 'Capacidad Molienda: Bajar Granulometría en residuos cerámicos', 'SP', 'Service', null, 'Want', '3000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('568', 'Spain', 'C', '61', 'Inorganic Calcium compounds', 'Inorgánicos Compuestos cálcicos', 'SP', 'Material', '01 01 02', 'Have', '1000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('569', 'Spain', 'C', '61', 'Inorganic products of acid treatment', 'Inorgánico Producto tratamiento gases ácidos', 'SP', 'Material', '06 13 99', 'Have', '99', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('570', 'Spain', 'C', '61', 'OSHAS 18001 Experience', 'Experiencia OSHAS 18001', 'SP', 'Service', '06 13 99', 'Have', '99', 'Hours per year', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('571', 'Spain', 'C', '61', 'Mineral Clay', 'Mineral Arcilla', 'SP', 'Material', '01 01 02', 'Have', '1000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('572', 'Spain', 'C', '61', 'Logistics Transportation: Towards the center', 'Logística Transporte: Hacia el centro', 'SP', 'Service', '01 01 02  ', 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('573', 'Spain', 'C', '61', 'CO2', 'CO2', 'SP', 'Material', null, 'Have', '50000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('574', 'Spain', 'C', '61', 'Grinding Capacity: Milling and micronization of materials', 'Capacidad Molienda: Molienda y micronización de materiales', 'SP', 'Service', null, 'Want', '10000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('575', 'Spain', 'C', '61', 'Bagging Capacity: Bagging', 'Capacidad Ensacar: Ensacado', 'SP', 'Service', null, 'Want', '1000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('576', 'Spain', 'C', '61', 'Minerals Dolostone:  High Purity (Mineral)', 'Minerales Dolomia: Dolomia Alta Pureza (Mineral)', 'SP', 'Material', null, 'Want', '99', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('577', 'Spain', 'C', '61', 'Thermal Energy', 'Energía Térmica: Energía Térmica', 'SP', 'Energy', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('578', 'Spain', 'C', '61', 'Logistics Transport from Madrid to Catalonia. Freight in Madrid destination: Catalonia, Aragon, Galicia', 'Logística Transporte: Transporte desde Madrid a Cataluña. Carga en Madrid destino: Cataluña, Aragón, Galicia', 'SP', 'Service', null, 'Want', '1000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('579', 'Spain', 'C', '62', 'Capacity Earth moving machinery: Heavy earthmoving machinery', 'Capacidad Maquinaria movimiento de tierras: Maquinaria pesada para movimiento de tierras', 'SP', 'Service', '01 01 02', 'Have', '2', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('580', 'Spain', 'C', '62', 'Land Capacity: Land for storage', 'Capacidad Terreno: Terrenos para acopios', 'SP', 'Service', '01 01 02', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('581', 'Spain', 'C', '62', 'Feldspar Mineral Arena: Humidity: 20%', 'Mineral Arena feldespática: Humedad: 20%', 'SP', 'Material', '01 01 02', 'Have', '15000', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('582', 'Spain', 'C', '62', 'Capacity Laboratory: Chemical laboratory, ceramic raw materials and quartz. Weekly scans', 'Capacidad Laboratorio: Laboratorio químico, materias primas cerámicas y cuarzo. Análisis semanales', 'SP', 'Service', null, 'Want', '36', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('583', 'Spain', 'C', '62', 'Energy Gas: Gas', 'Energía Gas: Gas', 'SP', 'Energy', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('584', 'Spain', 'C', '62', 'Logistics Transportation', 'Logística Transporte: (reporte)', 'SP', 'Service', '06 13 99', 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('585', 'Spain', 'C', '63', 'Inorganic Cement ', 'Inorgánico Cemento - Materiales: cemento', 'SP', 'Material', '17 01 01', 'Have', '5', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('586', 'Spain', 'C', '63', 'Mineral Aggregates', 'Mineral Áridos - ', 'SP', 'Material', '17 05 04', 'Have', '1', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('587', 'Spain', 'C', '63', 'Mineral Mortar ', 'Mineral Mortero - ', 'SP', 'Material', '17 01 07', 'Have', '22', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('588', 'Spain', 'C', '63', 'Capacity Organic waste treatment', 'Capacidad Tratamiento residuos orgánicos - ', 'SP', 'Service', '07 07 99', 'Have', '150', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('589', 'Spain', 'C', '63', 'Water', 'Agua - Agua', 'SP', 'Material', null, 'Want', '10', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('590', 'Spain', 'C', '63', 'Organic Waste - Organic Waste Decomposing', 'Orgánicos Residuos - Residuos Orgánicos que descompongan', 'SP', 'Material', null, 'Want', '200', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('591', 'Spain', 'C', '64', 'Ceramic Refractory - Remains of refractory material', 'Ceramica Refractario - Restos de material refractario', 'SP', 'Material', '06 13 99', 'Have', '500', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('592', 'Spain', 'C', '64', 'WEE - Electronic Waste', 'WEE - Residuos de Equipos electrónicos', 'SP', 'Material', '06 04 99', 'Have', '0', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('593', 'Spain', 'C', '64', 'Ceramics colored Remains ', 'Cerámica Restos Barredora - Restos de Barredora coloreados', 'SP', 'Material', '06 11 99', 'Have', '100', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('594', 'Spain', 'C', '64', 'Paper waste', 'Papel - Residuos de papel', 'SP', 'Material', null, 'Have', '100', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('595', 'Spain', 'C', '64', 'Experience in Integrated Environmental Authorization', 'Experiencia AAI - Experiencia en Autorización Ambiental Integrada', 'SP', 'Service', null, 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('596', 'Spain', 'C', '64', 'Plastic Waste', 'Plástico Residuos - Residuos de plástico', 'SP', 'Material', '15 01 02', 'Have', '10', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('597', 'Spain', 'C', '64', 'REACH Experience - REACH / CLP Experience', 'Experiencia REACH - Experiencia en REACH/CLP', 'SP', 'Service', null, 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('598', 'Spain', 'C', '64', 'Experience in standards and certification: ISO9000 / 14000 Standards', 'Experiencia Normativa - Experiencia en Normativa ISO9000/14000', 'SP', 'Service', null, 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('599', 'Spain', 'C', '64', 'Depurated Water - Processed Water', 'Agua Depurada - Agua de Proceso depurada', 'SP', 'Material', '19 08 99', 'Have', '2000', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('600', 'Spain', 'C', '64', 'Wood Pallets - Pallets in good condition but colored', 'Madera Pallets - Pallets en buen estado pero coloreados', 'SP', 'Tools', '03 01 99  ', 'Have', '1200', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('601', 'Spain', 'C', '64', 'Metals Waste - Waste containing metals', 'Metales Residuos - Residuos que contengan metales', 'SP', 'Material', null, 'Want', '99', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('602', 'Spain', 'C', '64', 'Experience in depuration - Experience in purification systems, gaseous pollutants', 'Experiencia Depuración - Experiencia en sistemas de depuración, contaminantes gaseosos', 'SP', 'Service', null, 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('603', 'Spain', 'C', '64', 'OHSAS Experience - OHSAS Experience', 'Experiencia OHSAS - Experiencia en OHSAS', 'SP', 'Service', null, 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('604', 'Spain', 'C', '65', 'Wood Pallets', 'Madera Pallets - Pallets de madera', 'SP', 'Tools', null, 'Have', '33655', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('605', 'Spain', 'C', '65', 'Logistics Transportation', 'Logística Transporte - (reporte)', 'SP', 'Service', '06 13 99', 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('606', 'Spain', 'C', '65', 'Plastic - Used plastic waste', 'Plástico - Residuo de plástico usado', 'SP', 'Material', '06 13 99', 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('607', 'Spain', 'C', '65', 'used organic rubber ', 'Orgánico Goma usada - ', 'SP', 'Material', '06 13 99', 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('608', 'Spain', 'C', '65', 'Used Big Bags - Dirty Packaging', 'Envase Big Bags usados - Sucios', 'SP', 'Tools', '06 13 99', 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('609', 'Spain', 'C', '65', 'Remains of ceramic frits', 'Frita Residuo - Restos de fritas cerámicas', 'SP', 'Material', '06 13 99', 'Want', '200', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('610', 'Spain', 'C', '65', 'Inorganic Alumina - Alumina Residues', 'Inorgánicos Alúmina - Residuos de alúmina', 'SP', 'Material', null, 'Want', '600', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('611', 'Spain', 'C', '65', 'Inorganic Residues deflocculant - Residues from the manufacture of deflocculants', 'Inorgánicos Residuo desfloculante - Residuos de la fabricación de desfloculantes', 'SP', 'Material', null, 'Want', '99', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('612', 'Spain', 'C', '65', 'Land warehouse covered indoor ', 'Terreno Almacén Cubierto - Almacén Cubierto', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('613', 'Spain', 'C', '66', 'Coloring Powder paint', 'Colorante Pintura en polvo - ', 'SP', 'Material', '08 01 99', 'Have', '100', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('614', 'Spain', 'C', '66', 'Water Sewage Sludge ', 'Agua Lodos de depuradora - ', 'SP', 'Material', '08 01 99', 'Have', '400', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('615', 'Spain', 'C', '66', 'Wood', 'Madera - ', 'SP', 'Material', '08 01 99', 'Have', '10', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('616', 'Spain', 'C', '66', 'Plastic Polyester', 'Plástico Poliéster - ', 'SP', 'Material', '08 01 99', 'Have', '120', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('617', 'Spain', 'C', '66', 'Carton Cardboard', 'Envase Cartón - ', 'SP', 'Material', '08 01 99', 'Have', '30', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('618', 'Spain', 'C', '66', 'Wood Tables - Wood in the form of boards', 'Madera Tablas - Madera en forma de tablas', 'SP', 'Material', null, 'Want', '10', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('619', 'Spain', 'C', '66', 'Metals Aluminum - Aluminum', 'Metales Aluminio - Aluminio', 'SP', 'Material', null, 'Want', '200', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('620', 'Spain', 'C', '66', 'Energy Gas - Propane Gas', 'Energía Gas - Gas Propano', 'SP', 'Energy', null, 'Want', '1', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('621', 'Spain', 'C', '67', 'FRX Capacity', 'Capacidad FRX - ', 'SP', 'Service', '06 13 99', 'Have', '2', 'Hours per day', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('622', 'Spain', 'C', '67', 'Enamel debris - In liquid', 'Esmalte Restos de esmalte - En líquido', 'SP', 'Material', '06 13 99', 'Have', '100', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('623', 'Spain', 'C', '67', 'Ceramic Tile Remainders - Crushed', 'Cerámica Restos baldosas - Triturados', 'SP', 'Material', '06 13 99', 'Have', '4800', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('624', 'Spain', 'C', '67', 'Ceramics Solid Grinding - Grinding of tiles', 'Cerámica Sólido de rectificado - Rectificado de baldosas', 'SP', 'Material', '06 13 99', 'Have', '600', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('625', 'Spain', 'C', '67', 'Process Water - Solids content: 5%', 'Agua De proceso - Contenido en sólidos: 5%', 'SP', 'Material', '06 13 99', 'Have', '10800', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('626', 'Spain', 'C', '67', 'Alternative energy to gas and electricity', 'Energía Alternativa - Energías alternativas al gas y a la electricidad', 'SP', 'Energy', null, 'Want', '35', 'kWh', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('627', 'Spain', 'C', '67', 'Inorganic Disflocculants - Disflocculants for the grinding of raw materials for the manufacture of tiles', 'Inorgánicos Desfloculantes - Desfloculantes para la molienda de materias primas para la fabricación de baldosas', 'SP', 'Material', null, 'Want', '600', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('628', 'Spain', 'C', '68', 'Land in the Port of Castellñon', 'Terreno Puerto - Terreno en el Puerto de Castellñon', 'SP', 'Service', null, 'Have', '2000000', 'Square Metres (m2)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('629', 'Spain', 'C', '68', 'Minerals Sweeps - Sweeps Bulk Solid Inertes', 'Minerales Barreduras - Barreduras Graneles Sólidos Inertes', 'SP', 'Material', '01 04 99', 'Have', '800', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('630', 'Spain', 'C', '68', 'Inorganic Bulbs - Traditional light bulbs by change to low consumption bulbs. Different powers', 'Inorgánicos Bombillas - Bombillas tradicionales por cambio a bombillas de bajo consumo. Distintas potencias', 'SP', 'Material', '06 13 99  ', 'Have', '300', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('631', 'Spain', 'C', '68', 'Inorganic construction waste', 'Inorgánicos Residuos Construcción - Materiales residuos de construcción', 'SP', 'Material', '06 13 99', 'Want', '5000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('632', 'Spain', 'C', '69', 'Grinding Capacity - Lower Granulometry in ceramic waste', 'Capacidad Molienda - Bajar Granulometría en residuos cerámicos', 'SP', 'Service', null, 'Have', '3000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('633', 'Spain', 'C', '69', 'Plastic', 'Plástico - ', 'SP', 'Material', null, 'Have', '10', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('634', 'Spain', 'C', '69', 'Energy Electricity - Nights, holidays, weekend August', 'Energía Electricidad - Noches, festivos, fin de semana Agosto', 'SP', 'Energy', '06 13 99', 'Have', '2000', 'MW', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('635', 'Spain', 'C', '69', 'Paper / Cardboard', 'Papel / Cartón - ', 'SP', 'Material', '06 13 99', 'Have', '15', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('636', 'Spain', 'C', '69', 'Wood Pallets broken - Wood waste: broken pallets', 'Madera Pallets rotos - Residuo de madera: pallets rotos', 'SP', 'Material', '06 13 99', 'Have', '12', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('637', 'Spain', 'C', '69', 'Cargo Logistics - Loading Containers', 'Logística Carga - Carga de Contenedores', 'SP', 'Tools', null, 'Want', '3', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('638', 'Spain', 'C', '69', 'Logistics Freight - Freight through freight', 'Logística Carga - Carga mediante carretila', 'SP', 'Service', null, 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('639', 'Spain', 'C', '69', 'Experience Automation - Experience in Process Automation', 'Experiencia Automatización - Experiencia en Automatización de Procesos', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('640', 'Spain', 'C', '69', 'Logistics Transport - Capacity Transport to Port and Cargo. 40 tonnes per week but in a timely manner', 'Logística Transporte - Capacidad Transporte a Puerto y Carga. 40 toneladas por semana pero de forma puntual', 'SP', 'Service', null, 'Want', '40000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('641', 'Spain', 'C', '69', 'Clean Water - Weekly Clean Water', 'Agua Limpia - Agua Limpia semanal', 'SP', 'Material', null, 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('642', 'Spain', 'C', '70', 'Empty containers - Waste empty containers (IBCs, Big Bags, ...)', 'Envases vacíos - Residuos de envases vacíos (GRG\'s, Big Bags, ...)', 'SP', 'Tools', '06 13 99  ', 'Have', '200', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('643', 'Spain', 'C', '70', 'Logistics Management - International Logistics Management', 'Logística Géstión - Gestión Logística Internacional', 'SP', 'Service', '06 13 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('644', 'Spain', 'C', '70', 'Experience certification ISO 9001, 14001 - Legal requirements', 'Experiencia ISO 9001, 14001 - Requisitos legales, ...', 'SP', 'Service', '06 13 99  ', 'Have', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('645', 'Spain', 'C', '70', 'Waste water - Purified or brine', 'Agua residual - Depurada o salmuera', 'SP', 'Material', '06 13 99  ', 'Have', '36500', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('646', 'Spain', 'C', '70', 'Inorganic Remains with Quartz - Remains with Quartz', 'Inorgánicos Restos con Cuarzo - Restos con Cuarzo', 'SP', 'Material', null, 'Want', '50000', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('647', 'Spain', 'C', '70', 'Inorganic Caustic Soda ', 'Inorgánicos Sosa Caústica - Sosa Caústica', 'SP', 'Material', null, 'Want', '2', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('648', 'Spain', 'C', '70', 'Experience Software Development - Software Development in environmental management, resources, etc', 'Experiencia Desarrollo Software - Desarrollo de Software en gestión ambiental, recursos, etc', 'SP', 'Service', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('649', 'Spain', 'C', '70', 'Electric Power - Waste drying (more than 100 Tons / year)', 'Energía Eléctrica - Secado de residuos (más de 100 Toneladas/año)', 'SP', 'Energy', null, 'Want', '99', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('650', 'Spain', 'C', '70', 'Legal Experience - Legal Requirements for Internationalization (Arab Countries, Africa, Russia)', 'Experiencia Legal - Requisitos legales para Internacionalización (Países Árabes, África, Rusia)', 'SP', 'Service', null, 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('651', 'Spain', 'C', '70', 'Experience Energy Efficiency - Experience in Energy Efficiency', 'Experiencia Eficiencia Energética - Experiencia en Eficiencia Energética', 'SP', 'Service', null, 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('652', 'Spain', 'C', '70', 'Glass Powder - Powder (medium size 1 mm)', 'Vidrio En polvo - En polvo (tamaño medio 1 mm)', 'SP', 'Material', '06 13 99', 'Want', '100', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('654', 'England', 'D', '71', 'Organics - Phenol - Waste phenol, quantity needs verifying (250MT per quarter - M=Million?)', 'Organics - Phenol - Waste phenol, quantity needs verifying (250MT per quarter - M=Million?)', 'EN', 'Material', '16 03 05', 'Have', '1000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('655', 'England', 'D', '71', 'Plastics - Drum and IBC waste - Quantities are variable', 'Plastics - Drum and IBC waste - Quantities are variable', 'EN', 'Tools', '15 01 05', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('656', 'England', 'D', '71', 'Organics - Mixed Solvent ', 'Organics - Mixed Solvent - ', 'EN', 'Material', '99 99 99', 'Have', '1000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('657', 'England', 'D', '71', 'Inorganics - Sodium Bromide Aqueous Solution - Two streams - one concentrated and one dilute - quantity to verify but tonnes', 'Inorganics - Sodium Bromide Aqueous Solution - Two streams - one concentrated and one dilute - quantity to verify but tonnes', 'EN', 'Material', '16 05 04', 'Have', '1', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('658', 'England', 'D', '71', 'Water - Aqueous Waste Streams - 200 - 5000 tonnes and continuous & batch supply of interest', 'Water - Aqueous Waste Streams - 200 - 5000 tonnes and continuous & batch supply of interest', 'EN', 'Material', '19 08 99', 'Want', '5000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('659', 'England', 'D', '71', 'Expertise - Advise of Good Manufacturing Practice - GMP for animal feed', 'Expertise - Advise of Good Manufacturing Practice - GMP for animal feed', 'EN', 'Service', '02 03 04', 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('660', 'England', 'D', '71', 'Capacity - Hazardouse Liquid Storage - 150 to 500 cubic metre tanks required on ad hoc basis', 'Capacity - Hazardouse Liquid Storage - 150 to 500 cubic metre tanks required on ad hoc basis', 'EN', 'Service', '99 99 99', 'Want', '500', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('661', 'England', 'D', '72', 'Packaging - Mixed waste - metals, wood, packaging mixed', 'Packaging - Mixed waste - metals, wood, packaging mixed', 'EN', 'Material', '15 01 06', 'Have', '1', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('662', 'England', 'D', '72', 'Water - Process Effluent - 5000 m3 per day', 'Water - Process Effluent - 5000 m3 per day', 'EN', 'Material', '19 08 99', 'Have', '1825000', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('663', 'England', 'D', '72', 'Organics - Hydrocarbon Wax - Quantity to verify - 10s to 100s of kilos - classed as Hazardous Waste', 'Organics - Hydrocarbon Wax - Quantity to verify - 10s to 100s of kilos - classed as Hazardous Waste', 'EN', 'Material', '07 06 04', 'Have', '0', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('664', 'England', 'D', '72', 'Organics - Hydrocarbon Tank Sludge - Quantity to verify - 1000s of m3 - periodic when tanks inspected and cleaned', 'Organics - Hydrocarbon Tank Sludge - Quantity to verify - 1000s of m3 - periodic when tanks inspected and cleaned', 'EN', 'Material', '07 06 12', 'Have', '1000', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('665', 'England', 'D', '72', 'Water - Rain Water - ', 'Water - Rain Water - ', 'EN', 'Material', '19 08 99', 'Want', '1000000', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('666', 'England', 'D', '72', 'Water - Potable Water - ', 'Water - Potable Water - ', 'EN', 'Material', '19 08 99', 'Want', '365000', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('667', 'England', 'D', '72', 'Energy - High Pressure Steam - ', 'Energy - High Pressure Steam - ', 'EN', 'Energy', '99 99 99', 'Want', '1314000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('668', 'England', 'D', '72', 'Energy - Electrical - ', 'Energy - Electrical - ', 'EN', 'Energy', '99 99 99', 'Want', '85000', 'MW', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('669', 'England', 'D', '73', 'Organics - Solid Residues - Non hazardous solid residues - 1 to 1.5 tonnes per hour - verify tonnage', 'Organics - Solid Residues - Non hazardous solid residues - 1 to 1.5 tonnes per hour - verify tonnage', 'EN', 'Material', '99 99 99', 'Have', '3500', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('670', 'England', 'D', '73', 'Water - Trade Effluent - 35 tonnes per hour', 'Water - Trade Effluent - 35 tonnes per hour', 'EN', 'Material', '19 08 99', 'Have', '84000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('671', 'England', 'D', '73', 'Wood - Waste Wood - ', 'Wood - Waste Wood - ', 'EN', 'Material', '03 01 01', 'Want', '300000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('672', 'England', 'D', '73', 'Organics - Carbon Rich Chemical Process Residues - ', 'Organics - Carbon Rich Chemical Process Residues - ', 'EN', 'Material', '99 99 99', 'Want', '100000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('673', 'England', 'D', '73', 'Energy - Refuse Derived Fuel - ', 'Energy - Refuse Derived Fuel - ', 'EN', 'Energy', '99 99 99', 'Want', '250000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('674', 'England', 'D', '73', 'Land - Sites for New Projects - ', 'Land - Sites for New Projects - ', 'EN', 'Material', '99 99 99', 'Want', '50000', 'Square Metres (m2)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('675', 'England', 'D', '73', 'Energy - Steam - 30GJ/hour required - assumed 24 x 300days and converted to KW', 'Energy - Steam - 30GJ/hour required - assumed 24 x 300days and converted to KW', 'EN', 'Energy', '99 99 99', 'Want', '60000000', 'kWh', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('676', 'England', 'D', '73', 'Plastics - Waste Plastics - ', 'Plastics - Waste Plastics - ', 'EN', 'Material', '02 01 04', 'Want', '200000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('677', 'England', 'D', '73', 'Experise - Project Partners - Waste chemical expertise required - converting waste to new products', 'Experise - Project Partners - Waste chemical expertise required - converting waste to new products', 'EN', 'Service', '99 99 99', 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('678', 'England', 'D', '74', 'Office cabins - office welfare cabins', 'Office cabins - office welfare cabins', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('679', 'England', 'D', '74', 'Dismantling engineering - dismantling demolition engineering', 'Dismantling engineering - dismantling demolition engineering', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('680', 'England', 'D', '74', 'ISO 9001 14001 - ', 'ISO 9001 14001 - ', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('681', 'England', 'D', '74', 'Capacity - Redundant Plant and Equipment - ', 'Capacity - Redundant Plant and Equipment - ', 'EN', 'Service', '99 99 99', 'Want', '6000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('682', 'England', 'D', '74', 'Expertise - Structural Engineering - Sporadic requirement', 'Expertise - Structural Engineering - Sporadic requirement', 'EN', 'Service', '99 99 99', 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('683', 'England', 'D', '74', 'Land - 2 Acres of Land - 2 acres required', 'Land - 2 Acres of Land - 2 acres required', 'EN', 'Service', '99 99 99', 'Want', '1', 'Hectares', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('684', 'England', 'D', '75', 'Organics - Acrylic Polymers - Soluble acrylic polymers - 30kte to 40kte year', 'Organics - Acrylic Polymers - Soluble acrylic polymers - 30kte to 40kte year', 'EN', 'Material', '99 99 99', 'Have', '40000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('685', 'England', 'D', '75', 'Organics - Ammonium Sulphate and Acrylic Polymers - Soluble in water', 'Organics - Ammonium Sulphate and Acrylic Polymers - Soluble in water', 'EN', 'Material', '99 99 99', 'Have', '12000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('686', 'England', 'D', '75', 'Energy - Electricity - ', 'Energy - Electricity - ', 'EN', 'Energy', '99 99 99', 'Have', '40', 'MW', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('687', 'England', 'D', '75', 'Land - Brownfield Site - 40 hectares', 'Land - Brownfield Site - 40 hectares', 'EN', 'Service', '99 99 99', 'Have', '40', 'Hectares', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('688', 'England', 'D', '75', 'Capacity - Office Space - Upper floor of building 15 offices', 'Capacity - Office Space - Upper floor of building 15 offices', 'EN', 'Service', '99 99 99', 'Have', '15', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('689', 'England', 'D', '75', 'Water - Demineralised Water - Variable quantities', 'Water - Demineralised Water - Variable quantities', 'EN', 'Material', '19 08 99', 'Have', '1', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('690', 'England', 'D', '75', 'Energy - Steam - Steam at various pressures - in variable quantities', 'Energy - Steam - Steam at various pressures - in variable quantities', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('691', 'England', 'D', '75', 'Inorganics - Hydrogen Cyanide - High purity', 'Inorganics - Hydrogen Cyanide - High purity', 'EN', 'Material', '06-03-11', 'Have', '5000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('692', 'England', 'D', '75', 'Storage - Bulk Trailers - 10 tonne capacity', 'Storage - Bulk Trailers - 10 tonne capacity', 'EN', 'Service', '99 99 99', 'Have', '10', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('693', 'England', 'D', '75', 'Energy - Steam - 30GJ/hour required - assumed 24 x 300days and converted to KW', 'Energy - Steam - 30GJ/hour required - assumed 24 x 300days and converted to KW', 'EN', 'Energy', '99 99 99', 'Have', '60000000', 'kWh', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('694', 'England', 'D', '75', 'Energy - High Pressure Steam - ', 'Energy - High Pressure Steam - ', 'EN', 'Energy', '99 99 99', 'Have', '1314000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('695', 'England', 'D', '75', 'Energy - Electrical - ', 'Energy - Electrical - ', 'EN', 'Energy', '99 99 99', 'Have', '85000', 'MW', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('696', 'England', 'D', '75', 'Energy - Electricity - 3064 MWh in the day and 1324 MWh at night', 'Energy - Electricity - 3064 MWh in the day and 1324 MWh at night', 'EN', 'Energy', '99 99 99', 'Have', '3100000000', 'kWh', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('697', 'England', 'D', '75', 'Energy - Low grade heat and steam - 10-100 te/hr of low grade heat input', 'Energy - Low grade heat and steam - 10-100 te/hr of low grade heat input', 'EN', 'Energy', '99 99 99', 'Have', '876000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('698', 'England', 'D', '76', 'Capacity - Tank Storage Capacity - 287m3 to 1000m3 tank storage capacity', 'Capacity - Tank Storage Capacity - 287m3 to 1000m3 tank storage capacity', 'EN', 'Service', '99 99 99', 'Have', '1000', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('699', 'England', 'D', '76', 'Capacity - Tank Storage Capacity - 1100m3 to 2000m3 (need to confirm)', 'Capacity - Tank Storage Capacity - 1100m3 to 2000m3 (need to confirm)', 'EN', 'Service', '99 99 99', 'Have', '2000', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('700', 'England', 'D', '76', 'Land - 15 acres of Land - Located in Billingham (Riverside)', 'Land - 15 acres of Land - Located in Billingham (Riverside)', 'EN', 'Service', '99 99 99', 'Have', '6', 'Hectares', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('701', 'England', 'D', '76', 'Capacity - Quayside Jetty - 4m draft previously used for solids imports', 'Capacity - Quayside Jetty - 4m draft previously used for solids imports', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('702', 'England', 'D', '76', 'Minerals - Sand - Seal sand available in short term 2-3 months', 'Minerals - Sand - Seal sand available in short term 2-3 months', 'EN', 'Material', '01 04 09', 'Have', '1', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('703', 'England', 'D', '76', 'Capacity - Jetty Access - ', 'Capacity - Jetty Access - ', 'EN', 'Service', '99 99 99', 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('704', 'England', 'D', '76', 'Food - Vegetable Oil - Waste vegetable oil', 'Food - Vegetable Oil - Waste vegetable oil', 'EN', 'Material', '05 01 06', 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('705', 'England', 'D', '76', 'Water - Water Supply - Post deregulation supply alternatives', 'Water - Water Supply - Post deregulation supply alternatives', 'EN', 'Material', '19 08 99', 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('706', 'England', 'D', '76', 'Energy - Electricity - 3064 MWh in the day and 1324 MWh at night', 'Energy - Electricity - 3064 MWh in the day and 1324 MWh at night', 'EN', 'Energy', '99 99 99', 'Want', '3100000000', 'kWh', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('707', 'England', 'D', '77', 'Energy - Infra Red Lighting - Requires 1 off large IR light', 'Energy - Infra Red Lighting - Requires 1 off large IR light', 'EN', 'Energy', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('708', 'England', 'D', '78', 'Mini plants engineering resource - ', 'Mini plants engineering resource - ', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('709', 'England', 'D', '78', 'Expertise in process chemistry catalysis - catalysis and catalysts', 'Expertise in process chemistry catalysis - catalysis and catalysts', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('710', 'England', 'D', '78', 'Inorganics - Transition Metal Salts - Cobalt, Nickel pure solid or solutions. 10 to >100 tonnes', 'Inorganics - Transition Metal Salts - Cobalt, Nickel pure solid or solutions. 10 to >100 tonnes', 'EN', 'Material', '06 03 13', 'Want', '100', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('711', 'England', 'D', '78', 'Expertise - Liquid Sulphur Processing - For proof of concept project', 'Expertise - Liquid Sulphur Processing - For proof of concept project', 'EN', 'Service', '99 99 99', 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('712', 'England', 'D', '79', 'Redundant equipment - Titanium tungsten lined vessels 120m3 to 50m3 volume', 'Redundant equipment - Titanium tungsten lined vessels 120m3 to 50m3 volume', 'EN', 'Tools', '99 99 99', 'Have', '10', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('713', 'England', 'D', '79', 'Acetaldehyde - Acetaldehyde Methyl Dioxolane', 'Acetaldehyde - Acetaldehyde Methyl Dioxolane', 'EN', 'Material', '99 99 99', 'Have', '900', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('714', 'England', 'D', '79', 'Land - brownfield serviced chemical site', 'Land - brownfield serviced chemical site', 'EN', 'Service', '99 99 99', 'Have', '100', 'Hectares', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('715', 'England', 'D', '79', 'Lubricating oils - ', 'Lubricating oils - ', 'EN', 'Material', '12 01 10', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('716', 'England', 'D', '79', 'Low grade heat condensate - 8MW per hour heat content', 'Low grade heat condensate - 8MW per hour heat content', 'EN', 'Energy', '99 99 99', 'Have', '70080', 'MW', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('717', 'England', 'D', '79', '30ft 20ft container liners - Polyethylene PE', '30ft 20ft container liners - Polyethylene PE', 'EN', 'Material', '02 01 04', 'Have', '10000', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('718', 'England', 'D', '79', 'Laboratory analysis - Gas chromatography XRF wet chemistry', 'Laboratory analysis - Gas chromatography XRF wet chemistry', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('719', 'England', 'D', '79', 'Block polymer PET - ', 'Block polymer PET - ', 'EN', 'Material', '02 01 04', 'Have', '240', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('720', 'England', 'D', '79', 'Capacity - 200m3/hr pipeline - 200m3 hr pipeline capacity from wilton site to bran sands', 'Capacity - 200m3/hr pipeline - 200m3 hr pipeline capacity from wilton site to bran sands', 'EN', 'Service', '99 99 99', 'Have', '1752000', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('721', 'England', 'D', '79', 'Energy - Heat - Heat input > 300 degC', 'Energy - Heat - Heat input > 300 degC', 'EN', 'Energy', '99 99 99', 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('722', 'England', 'D', '79', 'Food - Recycle Pet Food Grade Pellet - ', 'Food - Recycle Pet Food Grade Pellet - ', 'EN', 'Material', '02 02 99', 'Want', '50000', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('723', 'England', 'D', '80', 'grant funding - 60000 euros to help companies make innovative use of waste streams (£53000)', 'grant funding - 60000 euros to help companies make innovative use of waste streams (£53000)', 'EN', 'Service', '99 99 99', 'Have', '60000', 'Money (€)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('724', 'England', 'D', '81', 'Agriculture - Cattle Slurry - Quantity to verify, increases in Winter months', 'Agriculture - Cattle Slurry - Quantity to verify, increases in Winter months', 'EN', 'Material', '02 05 02', 'Have', '1', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('725', 'England', 'D', '81', 'Capacity - Water Treatment Capacity - Can be one off or continuous supply of liquid waste treatment capacity', 'Capacity - Water Treatment Capacity - Can be one off or continuous supply of liquid waste treatment capacity', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('726', 'England', 'D', '81', 'Agricultural - AD Digestate - ', 'Agricultural - AD Digestate - ', 'EN', 'Material', '19 06 06', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('727', 'England', 'D', '81', 'Capacity - Office accomodation and parking - ', 'Capacity - Office accomodation and parking - ', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('728', 'England', 'D', '81', 'Expertise - Water Purchasing Deals - Related opportunities from deregulation of water industry from April 2017 - UK wide', 'Expertise - Water Purchasing Deals - Related opportunities from deregulation of water industry from April 2017 - UK wide', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('729', 'England', 'D', '81', 'Expertise - Water Efficiency - water efficiency products and services e.g. audits AMR', 'Expertise - Water Efficiency - water efficiency products and services e.g. audits AMR', 'EN', 'Service', '99 99 99', 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('730', 'England', 'D', '81', 'Minerals - Sand - sand for cattle cubicles - monthly requirement', 'Minerals - Sand - sand for cattle cubicles - monthly requirement', 'EN', 'Material', '01 04 09', 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('731', 'England', 'D', '82', 'Water - Waste water non hazardous - 27te per day of process waste water', 'Water - Waste water non hazardous - 27te per day of process waste water', 'EN', 'Material', '19 08 99', 'Have', '9855', 'Cubic Metres (m3)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('732', 'England', 'D', '82', 'Expertise - Shared Metering Expertise - ', 'Expertise - Shared Metering Expertise - ', 'EN', 'Service', '99 99 99', 'Want', '8', 'Hours per day', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('733', 'England', 'D', '83', 'Expertise - Chemical Industry Services - Catalyst handling, tank cleaning etc.', 'Expertise - Chemical Industry Services - Catalyst handling, tank cleaning etc.', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('734', 'England', 'D', '83', 'Water - Aqueous waste ex silver processing - <1ppm silver, 2 loads per month and 20 tonnes per load', 'Water - Aqueous waste ex silver processing - <1ppm silver, 2 loads per month and 20 tonnes per load', 'EN', 'Material', '19 08 99', 'Have', '480', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('735', 'England', 'D', '83', 'Energy - Hazardous Waste for Energy - Must have CV greater than 11MJ/KJ', 'Energy - Hazardous Waste for Energy - Must have CV greater than 11MJ/KJ', 'EN', 'Energy', '99 99 99', 'Want', '5000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('736', 'England', 'D', '83', 'Metals - Silver Containing Wastes - Both liquid and solid - e.g. X-Ray films, photo chemicals. Any quantity considered.', 'Metals - Silver Containing Wastes - Both liquid and solid - e.g. X-Ray films, photo chemicals. Any quantity considered.', 'EN', 'Material', '10 07 99', 'Want', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('737', 'England', 'D', '84', 'Coal handling facility - Unit from old power station', 'Coal handling facility - Unit from old power station', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('738', 'England', 'D', '84', 'Warehouse space - 900m2 of warehouse space', 'Warehouse space - 900m2 of warehouse space', 'EN', 'Service', '99 99 99', 'Have', '900', 'Square Metres (m2)', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('739', 'England', 'D', '84', 'Offices - Laydown space office units', 'Offices - Laydown space office units', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('740', 'England', 'D', '84', 'Capacity - lab analytical services - trade effluent biomass analysis', 'Capacity - lab analytical services - trade effluent biomass analysis', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('741', 'England', 'D', '84', 'Land - green brownfield 700 acres - ', 'Land - green brownfield 700 acres - ', 'EN', 'Service', '99 99 99', 'Have', '280', 'Hectares', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('742', 'England', 'D', '84', 'Wood - Biomass - Biomass feedstock for powerstation', 'Wood - Biomass - Biomass feedstock for powerstation', 'EN', 'Energy', '17 02 01', 'Want', '300000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('743', 'England', 'D', '85', 'Minerals - Ash - Boiler, fly, APC ashes', 'Minerals - Ash - Boiler, fly, APC ashes', 'EN', 'Material', '10 01 01', 'Want', '3650', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('744', 'England', 'D', '85', 'Energy - Low grade heat and steam - 10-100 te/hr of low grade heat input', 'Energy - Low grade heat and steam - 10-100 te/hr of low grade heat input', 'EN', 'Energy', '99 99 99', 'Want', '876000', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('745', 'England', 'D', '86', 'Organic Chemicals - Mixed Solvent Waste - Located in Consett - exact quantity unknown but tonnes', 'Organic Chemicals - Mixed Solvent Waste - Located in Consett - exact quantity unknown but tonnes', 'EN', 'Material', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('746', 'England', 'D', '86', 'Inorganics - Sodium Formate - Sodium formate concentrated aqueous solution (with sodium sulphate and organics). Produced per campaign so difficult to quantify annually.', 'Inorganics - Sodium Formate - Sodium formate concentrated aqueous solution (with sodium sulphate and organics). Produced per campaign so difficult to quantify annually.', 'EN', 'Material', '06 01 06', 'Have', '10', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('747', 'England', 'D', '86', 'Expertise - Regulatory - Expertise in REACH and COMAH. Also ISO14001, ISO9001 and ISO18001', 'Expertise - Regulatory - Expertise in REACH and COMAH. Also ISO14001, ISO9001 and ISO18001', 'EN', 'Service', '99 99 99', 'Have', '1', 'Number', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('748', 'England', 'D', '86', 'Organics - Xylene waste water - Xylene waste containing water and agrochemicals', 'Organics - Xylene waste water - Xylene waste containing water and agrochemicals', 'EN', 'Material', '19 08 99', 'Have', '40', 'Tonnes', 'per year', null);
INSERT INTO `workshop_items2_full` VALUES ('749', 'England', 'D', '86', 'Capacity - Hydrogeneration Facilities - ', 'Capacity - Hydrogeneration Facilities - ', 'EN', 'Service', '99 99 99', 'Want', '1', 'Number', 'per year', null);

-- ----------------------------
-- Table structure for workshop_items2_subset3
-- ----------------------------
DROP TABLE IF EXISTS `workshop_items2_subset3`;
CREATE TABLE `workshop_items2_subset3` (
  `id` bigint(2) NOT NULL AUTO_INCREMENT,
  `Cluster` varchar(255) DEFAULT NULL,
  `Workshop` varchar(2) DEFAULT NULL,
  `Company_ID` bigint(20) DEFAULT NULL,
  `Waste_description` text,
  `waste_description_original` varchar(2048) DEFAULT NULL,
  `language` varchar(20) DEFAULT NULL,
  `Type` varchar(255) DEFAULT NULL,
  `Wastecode` varchar(255) DEFAULT NULL,
  `Have_want` varchar(255) DEFAULT NULL,
  `Quantity` bigint(20) DEFAULT NULL,
  `Measure_unit` varchar(255) DEFAULT NULL,
  `Frequency` varchar(255) DEFAULT NULL,
  `Remarks` text,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=750 DEFAULT CHARSET=latin1;

-- ----------------------------
-- Records of workshop_items2_subset3
-- ----------------------------
INSERT INTO `workshop_items2_subset3` VALUES ('3', 'Turkey', 'A', '2', 'plastic chips: Trimmed , broken scrap plastic', 'plastic chips: Trimmed , broken scrap plastic', 'EN', 'Material', '15 01 02', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_subset3` VALUES ('4', 'Turkey', 'A', '2', 'Pressed scrap metal', 'Pressed scrap metal', 'EN', 'Material', '15 01 04', 'Have', '5', 'tons', 'day', 'Ak Geri Dönü?üm: Can take non-hazardous metal waste;\nÖzvar Endüstriyel At?k: Can give to MKE;\nK?vanç Makine: They need steel scrap material as raw material in melting process;');
INSERT INTO `workshop_items2_subset3` VALUES ('5', 'Turkey', 'A', '2', 'Pressed scrap paper: Waste recycling facilities for paper mills or considered as intermediate products or raw materials', 'Pressed scrap paper: Waste recycling facilities for paper mills or considered as intermediate products or raw materials', 'EN', 'Material', '12 01 03', 'Have', '30', 'tons', 'day', 'ESOGÜ: Wastes as copper can be used for ceramic surface polishing (Doç. Dr. Çelik);');

-- ----------------------------
-- Table structure for workshop_items2_subset4
-- ----------------------------
DROP TABLE IF EXISTS `workshop_items2_subset4`;
CREATE TABLE `workshop_items2_subset4` (
  `id` bigint(2) NOT NULL AUTO_INCREMENT,
  `Cluster` varchar(255) DEFAULT NULL,
  `Workshop` varchar(2) DEFAULT NULL,
  `Company_ID` bigint(20) DEFAULT NULL,
  `Waste_description` text,
  `waste_description_original` varchar(2048) DEFAULT NULL,
  `language` varchar(20) DEFAULT NULL,
  `Type` varchar(255) DEFAULT NULL,
  `Wastecode` varchar(255) DEFAULT NULL,
  `Have_want` varchar(255) DEFAULT NULL,
  `Quantity` bigint(20) DEFAULT NULL,
  `Measure_unit` varchar(255) DEFAULT NULL,
  `Frequency` varchar(255) DEFAULT NULL,
  `Remarks` text,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=35 DEFAULT CHARSET=latin1;

-- ----------------------------
-- Records of workshop_items2_subset4
-- ----------------------------
INSERT INTO `workshop_items2_subset4` VALUES ('3', 'Turkey', 'A', '2', 'plastic chips: Trimmed , broken scrap plastic', 'plastic chips: Trimmed , broken scrap plastic', 'EN', 'Material', '15 01 02', 'Have', null, null, null, null);
INSERT INTO `workshop_items2_subset4` VALUES ('4', 'Turkey', 'A', '2', 'Electrical / electronic materials', 'Pressed scrap metal', 'EN', 'Material', '99 99 99', 'Have', '5', 'tons', 'day', 'Ak Geri Dönü?üm: Can take non-hazardous metal waste;\nÖzvar Endüstriyel At?k: Can give to MKE;\nK?vanç Makine: They need steel scrap material as raw material in melting process;');
INSERT INTO `workshop_items2_subset4` VALUES ('5', 'Turkey', 'A', '2', 'Pressed scrap paper: Waste recycling facilities for paper mills or considered as intermediate products or raw materials', 'Pressed scrap paper: Waste recycling facilities for paper mills or considered as intermediate products or raw materials', 'EN', 'Material', '12 01 03', 'Have', '30', 'tons', 'day', 'ESOGÜ: Wastes as copper can be used for ceramic surface polishing (Doç. Dr. Çelik);');
INSERT INTO `workshop_items2_subset4` VALUES ('34', 'Turkey', 'A', '4', 'Iron and steel scrap', 'Iron and steel scrap', 'EN', 'Material', '16 01 17', 'Have', '40', 'tons', 'year', 'An alternative fuel for use in cement production. Can take 1500-2000 tons / year capacity.');
SET FOREIGN_KEY_CHECKS=1;