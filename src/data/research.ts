export interface ResearchItem {
  title: string;
  description: string;
  tag: string;
  year: string;
  venue: string;
  href?: string;
}

export const research: ResearchItem[] = [
  {
    title: 'Toward Stochastic Realization Theory for Generalized Linear Switched Systems With Inputs: Decomposition Into Stochastic and Deterministic Components and Existence and Uniqueness of Innovation Form',
    description: 'We study a class of stochastic Generalized Linear Switched System (GLSS), which includes subclasses of jump-Markov, piecewise-linear and Linear Parameter-Varying (LPV) systems. We prove that the output of such systems can be decomposed into deterministic and stochastic components. Using this decomposition, we show existence of state-space representation in innovation form, and we provide sufficient conditions for such representations to be minimal and unique up to isomorphism.',
    tag: 'Journal Article',
    year: '2024',
    venue: 'IEEE Control Systems Letters',
    href: 'https://ieeexplore.ieee.org/document/10565856'
  },
  {
    title: 'Minimal covariance realization and system identification algorithm for a class of stochastic linear switched systems with i.i.d. switching',
    description: 'In this paper, we consider stochastic realization theory of Linear Switched Systems (LSS) with i.i.d. switching. We characterize minimality of stochastic LSSs and show existence and uniqueness (up to isomorphism) of minimal LSSs in innovation form. We present a realization algorithm to compute a minimal LSS in innovation form from output and input covariances. Finally, based on this realization algorithm, by replacing true covariances with empirical ones, we propose a statistically consistent system identification algorithm.',
    tag: 'Conference Paper',
    year: '2024',
    venue: 'IEEE Conference on Decision and Control',
    href: 'https://ieeexplore.ieee.org/document/10886432'
  },
  {
    title: 'Variable frequency monitoring of power grid: a modified observer approach',
    description: 'In most electrical networks, it can be strategic to monitor the frequency over the power grid. This monitoring can help in anticipating power dips on the network thus avoiding blackouts. The aim of this work is to monitor frequency changes with good dynamics. The proposed hybrid structure combines the advantages of a Phase Lock Loop with that of an adaptive observer. The PLL part offers a good tracking performance for slightly varying frequencies while the observer catches signal offset and significant changes in frequency. This paper gives an original technique to adjust frequency estimation dynamics. The tracking capability is validated by simulations and compared with other methods.',
    tag: 'Conference Paper',
    year: '2023',
    venue: 'IFAC World Congress',
    href: 'https://www.sciencedirect.com/science/article/pii/S2405896323022395'
  },
  {
    title: 'On minimal LPV state-space representations in innovation form: an algebraic characterization',
    description: 'In this paper we will propose a definition of the concept of minimal state-space representations in innovation form for LPV. We also present algebraic conditions for a stochastic LPV state-space representation to be minimal in forward innovation form and discuss an algorithm for transforming any stochastic LPV state-space representation to a minimal one in innovation form.',
    tag: 'Conference Paper',
    year: '2022',
    venue: 'IEEE Conference on Decision and Control',
    href: 'https://www.researchgate.net/publication/359574788_On_minimal_LPV_state-space_representations_in_innovation_form_an_algebraic_characterization',
  },
];
