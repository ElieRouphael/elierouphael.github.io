export interface ResumeEntry {
  title: string;
  org?: string;
  year?: string;
  body?: string;
}

export const sidebar = {
  name: 'Elie<br>Rouphael',
  subtitle: 'Senior Lecturer',
  contact: [
    'University of Lille',
    'Department of CS &amp; Automation',
    "Cit\u00e9 Scientifique, Villeneuve-d'Ascq",
    'elie.rouphael@univ-lille.fr',
  ],
  languages: ['French \u2014 Native', 'English \u2014 Fluent', 'Arabic \u2014 Heritage'],
  interests: [
    'Chess (club player, Elo 1850)',
    'Photography',
    'Classical music',
    'Cycling',
  ],
};

export const positions: ResumeEntry[] = [
  {
    title: 'Senior Lecturer (Ma\u00eetre de Conf\u00e9rences)',
    year: '2019 \u2014 Present',
    org: 'University of Lille, Department of Computer Science &amp; Automation',
    body:
      'Teaching and research in automatic control, embedded systems, and autonomous robotics. Responsible for the Real-Time C Programming and Sequential Logic Design courses at licence and master levels.',
  },
  {
    title: 'Postdoctoral Researcher',
    year: '2017 \u2014 2019',
    org: 'CRISTAL Laboratory (UMR CNRS 9189), University of Lille',
    body:
      'Research on formal verification of hybrid systems and co-design methods for safety-critical embedded controllers. Project funded by ANR under the CPS-Safety grant.',
  },
  {
    title: 'Teaching Assistant (ATER)',
    year: '2015 \u2014 2017',
    org: 'University of Lille, IUT A',
    body:
      'Full-time teaching assistant position covering digital electronics, C programming, and industrial automation. 192 hours per year.',
  },
];

export const education: ResumeEntry[] = [
  {
    title: 'PhD in Automatic Control and Robotics',
    year: '2012 \u2014 2015',
    org: 'University of Lille',
    body:
      'Thesis: <em>Event-Triggered Control for Networked Mobile Robot Formations.</em> Supervised by Prof. Jean-Pierre Barbot and Dr. Lotfi Belkoura. Mention tr\u00e8s honorable avec f\u00e9licitations du jury.',
  },
  {
    title: "Master's Degree in Embedded Systems Engineering",
    year: '2010 \u2014 2012',
    org: 'University of Lille, Polytech Lille',
    body:
      'Specialisation in real-time systems, digital signal processing, and FPGA design. Graduated with distinction (mention bien).',
  },
  {
    title: "Bachelor's Degree in Electrical Engineering &amp; Automation",
    year: '2007 \u2014 2010',
    org: 'University of Lille',
    body:
      'Foundations in circuit theory, control systems, and digital electronics. Class rank: 3rd out of 87.',
  },
];

export const teaching: ResumeEntry[] = [
  {
    title: 'Real-Time C Programming',
    org: 'Master 1, University of Lille \u2014 40h/year',
    body:
      'POSIX threads, real-time scheduling (EDF, RM), RTOS fundamentals, interrupt handling, and memory-mapped I/O on ARM Cortex-M targets.',
  },
  {
    title: 'Sequential Logic Design &amp; PLCs',
    org: 'Licence 3, University of Lille \u2014 48h/year',
    body:
      'Grafcet, IEC 61131-3 structured text and ladder logic, Petri net analysis, and industrial automation case studies.',
  },
  {
    title: 'Control Systems',
    org: 'Licence 2\u20133, University of Lille \u2014 60h/year',
    body:
      'Classical and state-space control, stability theory, frequency-domain methods, root locus, and digital implementation via z-transform.',
  },
  {
    title: 'Autonomous Robotics Project',
    org: 'Master 2, University of Lille \u2014 Supervision of student groups',
    body:
      'Year-long applied project developing perception, planning, and control modules for mobile robot platforms (ROS 2, Python, C++).',
  },
];

export const skills: string[] = [
  'C / C++',
  'Python',
  'MATLAB / Simulink',
  'VHDL',
  'ROS 2',
  'UPPAAL',
  'SPIN / Promela',
  'LaTeX',
  'Git',
  'FreeRTOS',
  'ARM Cortex-M',
  'FPGA (Xilinx)',
  'Linux',
  'CMake',
];

export const service: ResumeEntry[] = [
  {
    title: 'Associate Editor',
    year: '2022 \u2014 Present',
    org: 'IEEE Control Systems Letters',
  },
  {
    title: 'Programme Committee Member',
    year: '2021 \u2014 Present',
    org: 'IFAC World Congress; IEEE CDC; ECC',
  },
  {
    title: 'Director of Studies, M1 Embedded Systems',
    year: '2021 \u2014 Present',
    org: 'University of Lille',
  },
];
