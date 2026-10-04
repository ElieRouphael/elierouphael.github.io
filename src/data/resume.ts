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
      <i>Advisors:</i> Prof. Lotfi Belkoura & Dr. Mihály Petreczky (CNRS) <br/>
      <i>Defense:</i> October 24, 2025 <br/>
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
    org: 'LabCom I-TireLab - GIPSA-lab (Grenoble INP - UGA) / LIAS (Univ. Poitiers) / MICHELIN',
    body:
      `Working under the supervision of Prof. Guillaume Mercère and Prof. John-Jairo Martinez-Molina on real-time prediction of slippery road scenarios using hybrid modeling and machine learning, with the goal of improving vehicle safety.<br/>
      - Developing physics-informed and data-driven models for tire-road interaction and vehicle dynamics.<br/>
      - Combining Kalman filtering with data-based parameter estimation for real-time state and parameter estimation.<br/>
      - Building digital-twin-based methods for tire-road interaction modeling and dynamics prediction.`,
  },
  {
    title: 'PhD Researcher',
    year: 'Sep 2021 - Oct 2025',
    org: 'University of Lille - CRIStAL Laboratory',
    body:
      `- Developed stochastic realization results for generalized linear switched systems with inputs, including deterministic/stochastic decompositions and innovation-form representations.<br/>
      - Designed system identification and realization algorithms accounting for dynamical switching, with numerical validation procedures.<br/>
      - Worked with SZTAKI (Budapest) on deep state-space and time series modeling with generalization guarantees.<br/>`,
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

export interface Certificate {
  name: string;
  /** Public verification page (Coursera, Credly, ...). Without it the name shows as plain text. */
  url?: string;
}

export interface CertificationGroup {
  issuer: string;
  items: Certificate[];
}

const coursera = (id: string) => `https://www.coursera.org/account/accomplishments/verify/${id}`;

export const certifications: CertificationGroup[] = [
  {
    issuer: 'University of Michigan',
    items: [{ name: 'AI for Design and Optimization', url: coursera('2D5GB3BJHFXB') }],
  },
  {
    issuer: 'University of California, Santa Cruz',
    items: [{ name: 'Bayesian Statistics: From Concept to Data Analysis', url: coursera('I8P28MSPF616') }],
  },
  {
    issuer: 'DeepLearning.AI',
    items: [{ name: 'Neural Networks and Deep Learning', url: coursera('KYURTYL99Z7T') }],
  },
  {
    issuer: 'IBM',
    items: [
      { name: 'Deep Learning and Reinforcement Learning', url: coursera('GK0MYMKMM3KI') },
      { name: 'Specialized Models: Time Series and Survival Analysis', url: coursera('0K0FGSEZE355') },
      { name: 'Deep Learning with Keras and TensorFlow', url: coursera('NZJNT91GKH20') },
      { name: 'Introduction to Relational Databases (RDBMS)', url: coursera('DIAYZGXY5KQL') },
      { name: 'Introduction to DevOps', url: coursera('DF3EN2ZKCRY2') },
      { name: 'Introduction to Deep Learning & Neural Networks with Keras', url: coursera('TBF14TQEYMQV') },
      { name: 'Python Project for Data Engineering', url: coursera('K38NIHNO4DWG') },
      { name: 'Python for Data Science, AI & Development', url: coursera('0XZF5ESFE0GM') },
      { name: 'Introduction to Data Engineering', url: coursera('X209C9OT9Q9T') },
      { name: 'Machine Learning with Python', url: coursera('SSK9CKVWJ3BS') },
      { name: 'Generative AI and LLMs: Architecture and Data Preparation', url: coursera('3Z6BM3POGK2Z') },
    ],
  },
  {
    issuer: 'EECI Course',
    items: [{ name: 'Sparsity & Big Data in Control/ML' }],
  },
  {
    issuer: 'Spring School',
    items: [{ name: 'Closed-loop and nonlinear system ID and Deep Learning' }],
  },
  {
    issuer: 'ACES Summer School',
    items: [{ name: '60h in modern control theory' }],
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

export const technicalSkills: ResumeEntry[] = [
  {
    title: 'Programming',
    body: 'Python, MATLAB/Simulink, C/C++, SQL, R, Git.',
  },
  {
    title: 'Libraries',
    body: 'PyTorch, TensorFlow, Keras, JAX, Scikit-learn, Pandas, NumPy.',
  },
  {
    title: 'Teaching (labs, projects)',
    body: 'Python, C, Control Theory, Systems &Signals, Machine Learning.',
  },
  {
    title: 'Statistical Tools',
    body: 'Time series statistics, probabilistic modeling, Bayesian inference, Kalman filtering.',
  },
  {
    title: 'Expertise',
    body:
      'System Identification, Hybrid & Switched Systems, State-Space Modeling, Time Series Forecasting, Uncertainty Modeling, Deep Learning.',
  },
];

export const references: ResumeEntry[] = [
  {
    title: 'Prof. Lotfi Belkoura',
    org: 'Professor of System Identification, University of Lille',
    body: 'Email: lotfi.belkoura@univ-lille.fr',
  },
  {
    title: 'Dr. Mihaly Petreczky',
    org: 'HDR, CNRS Researcher, Centrale Lille',
    body: 'Email: mihaly.petreczky@centralelille.fr',
  },
  {
    title: 'Prof. Guillaume Mercère',
    org: 'Professor, LIAS, University of Poitiers',
    body: 'Email: guillaume.mercere@univ-poitiers.fr',
  },
  {
    title: 'Prof. John-Jairo Martinez-Molina',
    org: 'Professor, GIPSA-lab, Grenoble',
    body: 'Email: john-jairo.martinez-molina@grenoble-inp.fr',
  },
];
