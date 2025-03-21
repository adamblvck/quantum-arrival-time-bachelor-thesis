/*****************************************************
 * Airy Ai function for real x
 *  - piecewise: polynomial approximation on [-4,4]
 *  - asymptotic expansions for |x| > 4
 *****************************************************/

export function airyAi(x) {
	// If large positive x, use the decaying asymptotic expansion.
	if (x > 4) {
	  return airyAiAsymptoticPos(x);
	}
	// If large negative x, use the oscillatory asymptotic expansion.
	if (x < -4) {
	  return airyAiAsymptoticNeg(x);
	}
	// Otherwise, use the polynomial approximation on [-4,4].
	return airyAiPolyApprox(x);
}
  
  
  //------------------------
  // 2.1 Polynomial on [-4,4]
  //------------------------
  /*
	 Polynomial coefficients for Ai(x) on [-4,4].
	 This is a direct polynomial in x (NOT Chebyshev form).
	 If you want to check or regenerate, see a small script
	 in Python (e.g. using a least-squares or Chebyshev fit).
  */
  const aiCoeffs = [
	+0.35502805388781723926,       // a0
	-0.25881940379207215522,       // a1
	+0.02031783951018026277,       // a2
	-0.00134685118407398215,       // a3
	+0.00005411556638023885,       // a4
	-1.4927103941834672e-6,        // a5
	+2.6697102114112487e-8,        // a6
	-3.236181273817949e-10,        // a7
	+2.861834324030963e-12,        // a8
	-1.7793594301582907e-14,       // a9
	+8.129311916237573e-17,        // a10
	-2.6189451309295115e-19,       // a11
	+6.249064260261695e-22         // a12
  ];
  
  function airyAiPolyApprox(x) {
	// Evaluate polynomial via Horner’s scheme for efficiency
	// p(x) = a0 + x(a1 + x(a2 + ... ))
	let p = aiCoeffs[aiCoeffs.length - 1];
	for (let k = aiCoeffs.length - 2; k >= 0; --k) {
	  p = aiCoeffs[k] + x * p;
	}
	return p;
  }
  
  
  //-------------------------------------
  // 2.2 Asymptotic for large positive x
  //-------------------------------------
  function airyAiAsymptoticPos(x) {
	// Leading factor
	const sqrtPI = Math.sqrt(Math.PI);
	const x32 = Math.pow(x, 1.5); // x^(3/2)
	let front = 0.5 / sqrtPI * Math.pow(x, -0.25) * Math.exp(- (2/3) * x32);
  
	// 1 - 5/(48x^(3/2)) + ...
	// For large x, x^(3/2) is big; use a small correction
	const invX32 = 1 / x32;
	const corr = 1 - (5/48) * invX32; // you can add another term if you like
	return front * corr;
  }
  
  
  //-------------------------------------
  // 2.3 Asymptotic for large negative x
  //-------------------------------------
  function airyAiAsymptoticNeg(x) {
	const absx = -x;  // x<0, so absx = -x
	const sqrtPI = Math.sqrt(Math.PI);
	const x32 = Math.pow(absx, 1.5); // |x|^(3/2)
  
	// Leading amplitude
	let amp = 1 / sqrtPI * Math.pow(absx, -0.25);
  
	// Phase: (2/3)*|x|^(3/2) + pi/4
	let phase = (2/3)*x32 + Math.PI/4;
  
	// 1 + 5/(48|x|^(3/2)) + ...
	const invX32 = 1 / x32;
	const corr = 1 + (5/48) * invX32; // add more terms if needed
  
	// Ai ~ amplitude * sin(phase) * correction
	return amp * Math.sin(phase) * corr;
  }
  