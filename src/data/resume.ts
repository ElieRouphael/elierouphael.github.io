export interface ResumeEntry {
  title: string;
  org?: string;
  year?: string;
  body?: string;
}

export const sidebar = {
  name: 'Elie<br>Rouphael',
  subtitle: 'Postdoctoral Researcher in AI, System Identification & Control',
  contact: [
    'elie.rouphael.98@gmail.com',
    'github.com/ElieRouphael',
    'linkedin.com/in/elie-rouphael',
    '+33 6 04 08 36 44',
    'Grenoble, France',
  ],
  languages: ['Arabic (Native)', 'French (DELF - B2)', 'English (TOEIC 865 / 990)'],
  interests: [],
};

export const education: ResumeEntry[] = [
  {
    title: 'PhD in Automatic Control and Computer Science',
    year: '2021 - 2025',
    org: 'University of Lille - CRIStAL Laboratory, France',
    body:
      `
      <i>Thesis:</i> Towards Stochastic Realization Theory for Linear Switched Models <br/>
      <i>Focus:</i> Stochastic hybrid system identification, minimal realizations, learning with uncertainty.
      `,
  },
  {
    title: "Master's in Automatic Control and Electrical Energy",
    year: '2020 - 2021',
    org: 'University of Poitiers - LIAS Laboratory, France',
    body:
      'Graduated top of class. Research internship on non-stationary sinusoidal disturbance estimation.',
  },
  {
    title: "Engineering Degree in Electrical and Industrial Control Systems",
    year: '2016 - 2021',
    org: 'Lebanese University - Faculty of Engineering, Lebanon',
  },
];

export const experience: ResumeEntry[] = [
  {
    title: 'Postdoctoral Researcher',
    year: 'Nov 2025 - Present',
    org: 'LabCom I-TireLab',
    body:
      'Working, with MICHELIN, on real-time prediction of slippery scenarios using hybrid modeling and machine learning, with the goal of improving vehicle safety. Developing digital-twin-based methods for tire-road interaction modeling and dynamics prediction.',
  },
  {
    title: 'PhD Researcher',
    year: 'Sep 2021 - Oct 2025',
    org: 'University of Lille - CRIStAL Laboratory',
    body:
      `- Developed a theoretical framework for stochastic realization of linear switched systems.<br/>
      - Designed system identification algorithms accounting for dynamical switching.<br/>
      - Worked with SZTAKI (Budapest) on deep time series modeling with generalization guarantees.<br/>`,
  },
  {
    title: 'ATER - Teaching and Research Associate',
    year: 'Sep 2024 - 2025',
    org: 'University of Lille',
    body:
      '195+ hours of teaching: Python, C, Control Theory, Microcontrollers, GRAFCET. Supervising 3 student internships and 1 bachelor\'s project in machine learning and simulation.',
  },
  {
    title: 'Teaching Assistant (DCACE)',
    year: 'Oct 2022 - Sep 2024',
    org: 'University of Lille',
    body:
      '128+ hours of tutorials and labs in automation, signal processing, and feedback systems.',
  },
  {
    title: 'Research Intern',
    year: 'Mar - Jul 2021',
    org: 'LIAS Laboratory, University of Poitiers',
    body:
      'Topic: Detection of non-stationary harmonic disturbances in electrical systems.',
  },
  {
    title: 'Engineering Intern',
    year: 'Jul - Aug 2019',
    org: 'Cimenterie Nationale, Lebanon',
    body:
      'Installation and commissioning of electrical control panels in industrial systems.',
  },
];

export const certifications: ResumeEntry[] = [
  {
    title: 'IBM Certificates',
    body:
      'Introduction to DevOps; Deep Learning with Keras; Data Engineering; Machine Learning with Python; Python for AI & Development; Python Project for Data Engineering.',
  },
  {
    title: 'UCSC',
    body: 'Bayesian Statistics.',
  },
  {
    title: 'DeepLearning.ai',
    body: 'Neural Networks and Deep Learning.',
  },
  {
    title: 'EECI Course',
    body: 'Sparsity &Big Data in Control/ML.',
  },
  {
    title: 'Spring School',
    body: 'Closed-loop and nonlinear system ID and Deep Learning.',
  },
  {
    title: 'ACES Summer School',
    body: '60h in modern control theory.',
  },
];

export const projects: ResumeEntry[] = [
  {
    title: 'PAC Bounds for Stable SSMs',
    body:
      'This project studies Probably Approximately Correct (PAC) generalization bounds for deep advanced structures of State Space Models (SSMs). Collaborating with the SZTAKI research institute in Budapest, the project aims to theoretically justify the generalization ability of these models when they are stable.',
  },
  {
    title: 'Comparison of Classical and Deep Learning vs SSMs Time Series Models',
    body:
      'This project benchmarks traditional models (e.g., ARIMA, ARMAX, RNNs, CNNs, and Transformers) against SSMs (e.g., MAMBA) for time series forecasting.',
  },
  {
    title: 'Transport Mode Prediction with Hauts-de-France Data',
    body:
      'This project predicts individual transport mode choices (e.g., car, train, bike) to optimize mobility planning. It improves economic efficiency and ecological impact by helping policymakers promote sustainable and cost-effective transportation strategies.',
  },
  {
    title: 'System Identification Toolbox',
    body:
      'An ongoing initiative to develop and contribute to an open-source toolbox for system identification, focusing on LPV, hybrid, and switched systems.',
  },
];

export const publications: ResumeEntry[] = [
  {
    title:
      'Rouphael, E., et al. Minimal covariance realization and system identification algorithm for a class of stochastic linear switched systems with i.i.d. switching.',
    org: 'IEEE CDC 2024.',
  },
  {
    title:
      'Rouphael, E., et al. Toward Stochastic Realization Theory for Generalized Linear Switched Systems With Inputs: Decomposition Into Stochastic and Deterministic Components and Existence and Uniqueness of Innovation Form.',
    org: 'IEEE L-CSS, 2024.',
  },
  {
    title:
      'Rouphael, E., et al. Variable frequency monitoring of power grid: a modified observer approach.',
    org: 'IFAC-PapersOnLine, 2023.',
  },
  {
    title:
      'Rouphael, E., et al. On minimal LPV state-space representations in innovation form: an algebraic characterization.',
    org: 'IEEE CDC 2022.',
  },
];

export const technicalSkills: ResumeEntry[] = [
  {
    title: 'Programming',
    body: 'Python, MATLAB, C/C++, SQL, R.',
  },
  {
    title: 'Libraries',
    body: 'PyTorch, TensorFlow, Keras, Scikit-learn, Pandas, NumPy.',
  },
  {
    title: 'Teaching (labs, projects)',
    body: 'Python, C, Control Theory, Systems &Signals, Machine Learning.',
  },
  {
    title: 'Statistical Tools',
    body: 'Time series statistics, probabilistic modeling, Bayesian inference.',
  },
  {
    title: 'Expertise',
    body:
      'System Identification, Hybrid Systems, Time Series Forecasting, Uncertainty Modeling, Deep Learning.',
  },
];

export const references: ResumeEntry[] = [
  {
    title: 'Prof. Lotfi Belkoura',
    org: 'Professor of System Identification, University of Lille',
    body: 'Email: lotfi.belkoura@univ-lille.fr<br/>Phone: +33 (0)3 20 33 77 86',
  },
  {
    title: 'Dr. Mihaly Petreczky',
    org: 'HDR, CNRS Researcher, Centrale Lille',
    body: 'Email: mihaly.petreczky@centralelille.fr<br/>Phone: +33 (0)6 98 62 37 34',
  },
];
