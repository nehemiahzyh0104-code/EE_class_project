# EE_class_project
A dustbin for the project. Upload your update data and file here.

咱是robotwriter
————————————————————————————————————————————————————————————————————————————
# ShanghaiTech University SI100B Project: RoboWriter
## Introduction
This is a README file for **ShanghaiTech University SI100B Project: RoboWriter**. This file explains the requirements and grading criteria of the project, including project implementations, project reports, and project presentations. Please read throughout this file and make sure you have a group member with Windows platform before starting the project.

## I. Project Implementation (45 points)
### Requirements
#### Basic
1. Robot Arm Model (2 points): The universal_robots_ur5e model in the newest version of code repo : https://github.com/MAYBERoboLab/SIST_SI100B_RoboWriter 
2. Start pose and Terminal pose (5 points): The robot arm can start at any pose but needs to stop at the joint configuration: $[0.0,-2.32,-1.38,-2.45,1.57,0.0]$ rad after finishing the writing.
3. Writing area (5 points): The writing area is a square at the plane z=0.1 m. This square is defined by its four vertices: $(0.5, 0.1, 0.1)$, $(0.5, 0.6, 0.1)$, $(-0.5, 0.6, 0.1)$, and $(-0.5, 0.1, 0.1)$ m.
4. Lifting the end-effector (5 points): The end-effector of the robot arm should lift between each stroke and each character.
5. Writing contents (10 points): The robot should write a student's name (can choose one in your group) in Chinese and PinYin, and the student's school ID in the writing area. Visualize the written scripts/strokes of the words in real time.
6. Interpolation Method (7 points): Use multiple different interpolation methods to mimic the real strokes.
7. Writing speed (3 points): The robot's writing speed should be set at a reasonable value so that the whole writing time (from the robot's start pose to the terminal pose) is less than 3 minutes.
8. Plot joint state curves (3 points): Plot all the joint states of the robot arm when wirtting one word, let it be your family name in Chinese. 
9. Demo video (5 points): Record a video to demonstrate the writing process, you can adjust the camera for a better view. The name of the video should be "Group Member Names + Final Project Demo" in MP4 format.

#### Bonus
1. Git (1 point): Use Git to manage your code, and you need to have at least 3 commits for each group member.
2. Writing arbitrary words (2 points): Given an arbitrary string, you need to write the words in the string with the robot arm. The string will only contain English letters, spaces, numbers, commas, and periods.
3. Writing on a sphere (3 points): Write words on an area of the inner surface of a sphere defined by: $(x-0)^2+(y-0.35)^2+(z-1.3)^2=0$ and $z\leq 0.1$.

### Grading Criteria
1. For basic requirements, if you will only earn the points from requirement 1 to $k (k=1,2,3,4,5,6,7,8)$ if you fail to finish requirement $k+1$. If you fail to finish multiple requirements, $k$ will be the smallest requirement index that was not completed.
2. For basic requirement 7, you will earn 3 points if you only use linear interpolation.
3. The bonus tasks can be completed in any combination


## II. Project Presentations (4 points)
### Requirements
#### I. PPT Guidelines.
1. Language: English
2. Front Page Content: Title, Names of Group Members, and Student IDs.
3. Core Content: Must include clear project execution results --Plots and Videos.
#### II.Time Allocation (5 Minutes Total)
1. 4 Minutes: PPT presentation 
2. 1 Minute: Q&A session.

### Grading Criteria
1. Time Management [15%]: Timing must be within a ±1 minute margin. 
2. Visual Clarity [10%]: Slides should have a clear layout that is professional and easy to read.
3. Logical Structure [60%]: The presentation should include Methodology, Results, Limitations, and Future Work/Possible Improvements.
4. Q&A Session [15%]: Ability to provide clear answers to questions.


## III. Project Reports (6 points)
### Requirements
1.  Project Overview: Briefly introduce the research background, research purpose, application scenarios and overall work content of the project.
2.  System Design: Describe the overall architecture of the system, module divisions, key technical solutions, and the relationship between each module.
3.  Implementation: Describe in detail the specific implementation process of the system or project, including the technologies, algorithms, tools and implementation steps used.
4.  Testing and Results: Introduce the testing method and test environment, and display the system operation results and performance analysis.
5.  Discussion and Improvement: Analyze the significance of the experimental results, existing problems in the system, and propose possible improvement directions.
6.  Conclusion: Summarize the work and results of the overall project and briefly look forward to future research or applications.
 
### Grading creteria
1.  Completeness(25%): Evaluate whether the report contains all required chapters, whether the content is complete, whether the structure is clear, and whether key parts are not missing.
2.  Technical Explanation(30%): Evaluate whether the explanations of system design, technical solutions and implementation processes are accurate and clear, and reflect understanding of relevant technologies.
3.  Results Analysis(25%): Evaluate whether the analysis of experimental results or system output is reasonable and whether it can be effectively explained in combination with data or phenomena.
4.  Reflection and Improvement(20%): Evaluate the depth of reflection on project shortcomings and whether the proposed improvements are feasible and meaningful.

# 上海科技大学 SI100B 项目：RoboWriter
## 引言
本文档为**上海科技大学 SI100B 项目：RoboWriter**的说明文件。本文档阐述了项目的需求与评分标准，包括项目实现、项目报告和项目展示。请完整阅读本文档，并在开始项目前确保您的小组中有一名成员使用 Windows 平台。

## I. 项目实现 (45 分)
### 要求
#### 基础要求
1.  机械臂模型 (2 分)：使用最新版本代码库中的 universal\_robots\_ur5e 模型，地址为：https://github.com/MAYBERoboLab/SIST_SI100B_RoboWriter
2.  起始位姿与终止位姿 (5 分)：机械臂可以从任意位姿开始，但完成书写后必须停止在关节配置：$[0.0,-2.32,-1.38,-2.45,1.57,0.0]$ 弧度。
3.  书写区域 (5 分)：书写区域是 z=0.1 m 平面上的一个正方形。该正方形由其四个顶点定义：$(0.5, 0.1, 0.1)$, $(0.5, 0.6, 0.1)$, $(-0.5, 0.6, 0.1)$, 和 $(-0.5, 0.1, 0.1)$ 米。
4.  抬笔 (5 分)：机器臂末端执行器应在每一笔划和每个字符之间抬起。
5.  书写内容 (10 分)：机器人应在书写区域内书写一名学生的中文姓名和拼音，以及该学生的学号。实时可视化文字的书写笔划。
6.  插值方法 (7 分)：使用多种不同的插值方法来模拟真实的笔划。
7.  书写速度 (3 分)：机器人的书写速度应设置为合理值，使得整个书写过程（从机器人起始位姿到终止位姿）总时间少于 3 分钟。
8.  绘制关节状态曲线 (3 分)：绘制机器臂在书写一个字（例如您的中文姓氏）时所有关节的状态曲线。
9.  演示视频 (5 分)：录制视频展示书写过程，可以调整摄像机角度以获得更佳视角。视频名称应为“小组成员姓名 + Final Project Demo”，格式为 MP4。

#### 附加任务
1.  Git (1 分)：使用 Git 管理代码，每个小组成员需至少有 3 次提交记录。
2.  书写任意单词 (2 分)：给定任意字符串，需要用机器臂书写其中的单词。字符串仅包含英文字母、空格、数字、逗号和句点。
3.  在球面上书写 (3 分)：在球体内表面的一块区域内书写文字，球面由方程：$(x-0)^2+(y-0.35)^2+(z-1.3)^2=0$ 定义，且满足 $z\leq 0.1$。

### 评分标准
1.  对于基础要求，只有当你完成了第 1 到 $k (k=1,2,3,4,5,6,7,8)$ 项要求时，才能获得这些项的分数。如果第 $k+1$ 项要求未完成，则只能获得前 $k$ 项的分数。如果多项要求未完成，$k$ 将是未能完成的最小要求序号。
2.  对于基础要求 7，如果只使用了线性插值，则只能获得 3 分中的 1 分。
3.  附加任务可以任意组合完成。

## II. 项目展示 (4 分)
### 要求
#### I. PPT 指南。
1.  语言：英语
2.  首页内容：标题、小组成员姓名及学号。
3.  核心内容：必须包含清晰的项目执行结果——图表和视频。
#### II. 时间分配 (总计 5 分钟)
1.  4 分钟：PPT 演示
2.  1 分钟：问答环节

### 评分标准
1.  时间管理 [15%]：时间必须在规定时间的 ±1 分钟内。
2.  视觉清晰度 [10%]：幻灯片布局应清晰、专业且易于阅读。
3.  逻辑结构 [60%]：演示内容应包括方法论、结果、局限性以及未来工作/可能的改进。
4.  问答环节 [15%]：能够清晰回答提问。

## III. 项目报告 (6 分)
### 要求
1.  项目概述：简要介绍项目的研究背景、研究目的、应用场景和总体工作内容。
2.  系统设计：描述系统的整体架构、模块划分、关键技术方案以及各模块间的关系。
3.  实现过程：详细描述系统或项目的具体实现过程，包括所使用的技术、算法、工具和实施步骤。
4.  测试与结果：介绍测试方法和测试环境，展示系统运行结果和性能分析。
5.  讨论与改进：分析实验结果的意义、系统中存在的问题，并提出可能的改进方向。
6.  结论：总结整个项目的工作与成果，并对未来的研究或应用进行简要展望。

### 评分标准
1.  完整性 (25%)：评估报告是否包含所有要求的章节，内容是否完整，结构是否清晰，关键部分是否缺失。
2.  技术阐述 (30%)：评估对系统设计、技术方案和实现过程的解释是否准确、清晰，并体现出对相关技术的理解。
3.  结果分析 (25%)：评估对实验结果或系统输出的分析是否合理，能否结合数据或现象进行有效解释。
4.  反思与改进 (20%)：评估对项目不足之处的反思深度，以及提出的改进措施是否可行且有意义。
