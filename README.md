# ML Training Reference Architectures & Tests <!-- omit from toc -->

This repository contains reference architectures and test cases for distributed model training with [AWS ParallelCluster](https://docs.aws.amazon.com/parallelcluster/latest/ug/what-is-aws-parallelcluster.html), [AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/what-is-batch.html), and [Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html). The test cases cover different types and sizes of models as well as different frameworks and parallel optimizations (Pytorch DDP/FSDP, MegatronLM, NemoMegatron...).

The major components of this directory are:

```bash
reference-architectures/
|-- 1.architectures/               # CloudFormation templates for reference arch
|-- 2.ami_and_containers/          # Scripts to create AMIs and container images
|-- 3.test_cases/                  # Reference test cases and/or benchmark scripts
|-- 3.validation_scripts/          # Tools to measure performance or troubleshoot
`-- ...
```

**NOTE**: the architectures are designed to work with the S3 bucket and VPC created using reference templates `1.architectures/0.s3/` and `1.architectures/1.vpc_network/`. _You're strongly recommended to deploy these two templates **before** deploying any of the reference architectures._

## 1. Architectures

Architectures are located in `1.architectures` and consists of utilities and service related architectures

| Name                    | Category | Usage                                               |
| ----------------------- | -------- | --------------------------------------------------- |
| `0.s3`                  | Storage  | Create an S3 bucket                                 |
| `1.vpc_network`         | Network  | Create a VPC with subnets required resources        |
| `2.aws-parallelcluster` | Compute  | Cluster templates for GPU & custom silicon training |
| `3.aws-batch`           | Compute  | AWS Batch template for distributed training         |
| `4.amazon-eks`          | Compute  | Manifest files to train with Amazon EKS         |

More will come, feel free to add new ones (EKS, Ray?)

## 2. Custom Amazon Machine Images

Custom machine images can be built using [Packer](www.packer.io) for AWS ParallelCluster, Amazon EKS and plain EC2. These images are based are on Ansible roles and playbooks.

## 3. Test cases: support matrix

All test cases are under `3.test_cases/`. You can go in each test case directory to learn how to run it.

| Test cases                 | Slurm | Kubernetes | AWS Batch  |
| -------------------------- | ----- | ---------- | ---------- |
| `1.megatron-lm`            |  ✅   | ❓          | ❓         |
| `2.nemo-launcher`          |  ✅   | ❌          | ❌         |
| `3.MPT`                    |  ✅   | ❓          | ❓         |
| `4.DDP`                    |  ✅   | ❓          | ❓         |
| `5.param-benchmark`        |  ✅   | ❓          | ❓         |
| `6.stable-diffusion`       |  ✅   | ❓          | ❓         |
| `7.tensorflow-distributed` |  ✅   | ❓          | ❓         |

## 4. Validation scripts

Utilities scripts and micro-benchmarks examples are set under `4.validation_scripts/`.

## 5. CI

Integration tests are written in [pytest](https://docs.pytest.org). Just run:

```bash
pytest .
```

Alternatively you can run tests with out capturing stdout and keeping all docker images an other artifacts.

```bash
pytest -s --keep-artifacts=t
```

## 6. Contributors

Thanks to all the contributors for building, reviewing and testing.

- Pierre-Yves Aquilanti - pierreya@
- Verdi March - marcverd@
- Uros Lipovsek - lipovsek@
- Keita Watanabe - mlkeita@
- Ankur Srivastava - awsankur@
- Alex Iankoulski - iankouls@
- Tom McDonald - tjm@
- Sean Smith - seaam@
- Jianying Lang - langjian@
- Maxime Hugues - maxhaws@
- Dmitry Monakhov dmonakhov@
