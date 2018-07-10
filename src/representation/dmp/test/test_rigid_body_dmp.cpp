#define PRIVATE public
#include <catch/catch.hpp>
#include <vector>
#include <iostream>
#include <string>
#include "test_helpers.h"
#include "RigidBodyDmpConfig.h"
#include "RigidBodyDmp.h"
#include "DMPModel.h"
#include <lib_manager/LibManager.hpp>

using namespace std;
using namespace dmp;
using namespace dmp_cpp;
using namespace Eigen;


DMPModel getModel()
{
  DMPModel model;
  model.cs_dt = 0.01;
  model.cs_execution_time = 5.0;
  model.model_name = "";
  model.rbf_centers = {1, 0.829601643194907, 0.68823888639169, 0.570964111061179, 0.473672764741673, 0.392959703966367, 0.326000016119882, 0.27045014905462, 0.22436588805802, 0.186134309409818, 0.154417328941334, 0.1281048698275, 0.106276010510163, 0.0881667529514306, 0.0731432831236663, 0.0606797878680638, 0.0503400517240641, 0.0417621896288002, 0.03464598113947, 0.028742362883404, 0.0238447114773763, 0.0197816118231398, 0.0164108576735206, 0.0136144744921904, 0.0112945904099563};
  model.rbf_widths = {79.3021357521556, 115.224724031393, 167.41966533669, 243.25807310081, 353.450056238711, 513.557230239902, 746.190371387455, 1084.20257288801, 1575.3288492202, 2288.9274064117, 3325.77459900897, 4832.29684455261, 7021.24936573615, 10201.7620691956, 14822.9957227229, 21537.573676543, 31293.7471311169, 45469.3097845492, 66066.1736551044, 95993.0846126371, 139476.403485294, 202656.964381283, 294457.300202516, 427841.707326799, 427841.707326799};
  model.ts_alpha_z = 25.0;
  model.ts_beta_z = 6.25;
  model.ts_dt = 0.01;
  model.ts_tau = model.cs_execution_time;
  //model.ft_weights = TestHelpers::random_matrix(6, model.rbf_centers.size());
  model.ft_weights = TestHelpers::matrix(6, model.rbf_centers.size(), 0);
  model.cs_alpha = 5.0;
  return model;
}

RigidBodyDmpConfig getConfig(const DMPModel& model)
{
  RigidBodyDmpConfig cfg;
  cfg.rotationConfig.config_name = "";
  cfg.translationConfig.config_name = "";
  cfg.rotationConfig.startVelocity = {0, 0, 0};
  cfg.rotationConfig.endPosition = {1, 0.1, 0.7, 13};
  cfg.rotationConfig.startPosition = {1, 2, 3, 4};
  cfg.rotationConfig.executionTime = model.cs_execution_time;
  cfg.rotationConfig.fullyInitialized = true;

  cfg.translationConfig.dmp_endAcceleration ={0, 0, 0};
  cfg.translationConfig.dmp_endPosition = {1, 2, 3};
  cfg.translationConfig.dmp_endVelocity = {0, 1, 2};
  cfg.translationConfig.dmp_startVelocity = {0, 0, 0};
  cfg.translationConfig.dmp_startAcceleration = {0, 0, 0};
  cfg.translationConfig.dmp_startPosition = {0, 0, 0};
  cfg.translationConfig.dmp_execution_time = model.cs_execution_time;
  cfg.translationConfig.fullyInitialized = true;
  cfg.fullyInitialized = true;
  return cfg;
}

TEST_CASE("simple run", "[RigidBodyDmp]")
{

  DMPModel model = getModel();
  REQUIRE(model.is_valid());
  RigidBodyDmpConfig cfg = getConfig(model);
  REQUIRE(cfg.isValid());

  lib_manager::LibManager manager;
  RigidBodyDmp dmp(&manager);
  REQUIRE(dmp.initialize(model));
  REQUIRE(dmp.configure(cfg));

  double data[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4};
  while(dmp.canStep())
  {
    dmp.setInputs(data, 13);
    dmp.step();
    dmp.getOutputs(data, 13);
  }
  REQUIRE(data[0] == Approx(cfg.translationConfig.dmp_endPosition[0]).epsilon(0.1));
  REQUIRE(data[1] == Approx(cfg.translationConfig.dmp_endPosition[1]).epsilon(0.1));
  REQUIRE(data[2] == Approx(cfg.translationConfig.dmp_endPosition[2]).epsilon(0.1));

  REQUIRE(data[3] == Approx(cfg.translationConfig.dmp_endVelocity[0]).epsilon(0.1));
  REQUIRE(data[4] == Approx(cfg.translationConfig.dmp_endVelocity[1]).epsilon(0.1));
  REQUIRE(data[5] == Approx(cfg.translationConfig.dmp_endVelocity[2]).epsilon(0.1));

  REQUIRE(data[6] == Approx(cfg.translationConfig.dmp_endAcceleration[0]).epsilon(0.1));
  REQUIRE(data[7] == Approx(cfg.translationConfig.dmp_endAcceleration[1]).epsilon(0.1));
  REQUIRE(data[8] == Approx(cfg.translationConfig.dmp_endAcceleration[2]).epsilon(0.1));

  Eigen::Quaterniond targetRot(cfg.rotationConfig.endPosition[0], cfg.rotationConfig.endPosition[1],
                               cfg.rotationConfig.endPosition[2], cfg.rotationConfig.endPosition[3]);
  targetRot.normalize();
  REQUIRE(data[9] == Approx(targetRot.w()).epsilon(0.1));
  REQUIRE(data[10] == Approx(targetRot.x()).epsilon(0.1));
  REQUIRE(data[11] == Approx(targetRot.y()).epsilon(0.1));
  REQUIRE(data[12] == Approx(targetRot.z()).epsilon(0.1));
}


TEST_CASE("set weights", "[RigidBodyDmp]")
{

  DMPModel model = getModel();
  REQUIRE(model.is_valid());
  RigidBodyDmpConfig cfg = getConfig(model);
  REQUIRE(cfg.isValid());

  lib_manager::LibManager manager;
  RigidBodyDmp dmp(&manager);
  REQUIRE(dmp.initialize(model));
  REQUIRE(dmp.configure(cfg));

  ArrayXXd weights = ArrayXXd::Random(6, model.rbf_centers.size());
  assert(model.rbf_centers.size() == 25); //because I am too lazy to malloc weightsArr
  double weightsArr[25][6];

  for(int i = 0; i < weights.cols(); ++i)
  {
    for(int j = 0; j < weights.rows(); ++j)
    {
      weightsArr[i][j] = weights.col(i)[j];
    }
  }

  dmp.setWeights(&weightsArr[0][0], 6, 25);

  for(int i = 0; i < weights.cols(); ++i)
  {
    for(int j = 0; j < 3; ++j)
    {
      const double w = dmp.translationDmp->weights(j, i);
      REQUIRE(w == weights(j, i));
    }
  }

  for(int i = 0; i < weights.cols(); ++i)
  {
    for(int j = 0; j < 3; ++j)
    {
      const double w = dmp.rotationDmp->weights(j, i);
      REQUIRE(w == weights(j + 3, i));
    }
  }



}
