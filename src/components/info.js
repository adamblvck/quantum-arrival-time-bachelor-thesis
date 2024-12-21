export const markdownContent = `
# Normal Curves and Their Closest Approaches

This text explains how and why we plot various surfaces, contours, and “normals distance” curves, both in **2D** (for a single-variable function) and extended to **3D** (for a function $z = f(x,y)$). The screenshot above shows a dashboard interface that visualizes these ideas, with parameters \$begin:math:text$(a,b)\\$end:math:text$ controlling how the second point on the curve/surface is selected, and a parameter $k$ representing thresholds or offsets for normal-distance analysis.

---

## 1. Gist of What We Want to Plot

We begin with a function $f$. Depending on the context:

1. **1D version**  
   - A function $y = f(x)$.  
   - At each point $x$, we draw the **normal** line to the curve.  
   - We pick a point at $x + a$ (or $x + k$) to form a second normal, and then measure or visualize how these two normal lines intersect or come close.  

2. **2D → 3D version**  
   - A function $z = f(x,y)$.  
   - At each $(x,y)$ in some domain, we consider the **normal vector** to the surface.  
   - Then we pick a second point $(x+a,\; y+b)$ and form a second normal.  
   - We can compute or plot:
     - The **surface** $f(x,y)$.  
     - The **distance** between these two normals at each point.  
     - The **closest approach** line segments (or midpoints) between the two normals in 3D space.

Essentially, we are drawing or analyzing:
- The original curve/surface.
- Partial derivatives (e.g., $f_x$, $f_y$).
- A separate surface or contour representing the minimal distance or intersection points of normal lines.

---

## 2. The “2-Analysis” (Single-Variable Function)

### 2.1 Core Equations

For **1D** $y = f(x)$:

1. **Point 1**:  
   $$
   P = \\bigl(x,\\; f(x)\\bigr).
   $$

2. **Point 2**:  
   $$
   Q = \\bigl(x + a,\\; f(x + a)\\bigr).
   $$

3. **Tangent Slopes**:  
   - slope at $x$ is $f'(x)$,  
   - slope at $(x+a)$ is $f'(x+a)$.

4. **Normal Slopes**:  
   $$
   m_1 = -\\frac{1}{f'(x)}, 
   \\quad
   m_2 = -\\frac{1}{f'(x+a)}.
   $$

5. **Lines**:  
   $$
   L_1:\\; y - f(x) = m_1\\,\\bigl(X - x\\bigr),
   \\quad
   L_2:\\; y - f(x+a) = m_2\\,\\bigl(X - (x + a)\\bigr).
   $$

6. **Intersection or Parallel**  
   - If $m_1 \\neq m_2$, the lines intersect uniquely (distance $= 0$).  
   - If $m_1 = m_2$, they are parallel; we compute the minimal distance between parallel lines.

A **“normal cross curve”** emerges by letting $x$ vary and capturing each intersection or closest midpoint. The parameter $a$ (or $k$) sets the gap along the $x$-axis between the two normal lines.

### 2.2 Significance of Parameters $a$ and $b$

- In the 1D case, $b$ is often zero, but we can generalize if desired.  
- $a$ (or $k$) is the **horizontal separation** along the $x$-axis for the second normal.

---

## 3. Extension to 3D ($z = f(x,y)$)

### 3.1 Core Equations

For **3D** $z = f(x,y)$:

1. **Point 1**:  
   $$
   P = \\bigl(x,\\; y,\\; f(x,y)\\bigr).
   $$

2. **Point 2**:  
   $$
   Q = \\bigl(x + a,\\; y + b,\\; f(x + a,\\; y + b)\\bigr).
   $$

3. **Surface Normals**:  
   $$
   \\mathbf{N}_1 = \\bigl(f_x(x,y),\\; f_y(x,y),\\; -1\\bigr), 
   \\quad
   \\mathbf{N}_2 = \\bigl(f_x(x+a,\\; y+b),\\; f_y(x+a,\\; y+b),\\; -1\\bigr).
   $$

4. **Normal Lines**:  
   $$
   L_1(s) = P + s\\,\\mathbf{N}_1, 
   \\quad
   L_2(t) = Q + t\\,\\mathbf{N}_2.
   $$

5. **Closest Approach**:  
   - Solve for the minimal distance between these two lines.  
   - Plot or store that distance as a **scalar field** $d(x,y)$.  
   - Optionally, find the midpoints of each minimal-distance segment to create another geometric visualization.

### 3.2 Significance of $(a,b)$ in 3D

- $(a,b)$ is the offset in the $(x,y)$ plane.  
- This means the second normal is drawn at $(x+a,\\; y+b)$, giving a two-parameter shift.  
- You can form a family of such normals for all $(x,y)$ in the domain, revealing interesting geometric behavior in 3D.

---

## 4. Usage Guide for the Dashboard Interface

Referencing the screenshot:

1. **Menu (Left Panel)**  
   - **Function f(x,y):** Enter your function (e.g., \`sin(x)*cos(y)\`, \`(x^2 + y^2)^(1/2)\`) in a parseable math syntax.  
   - **Enable Function:** Toggle to either show or hide the function-based plots.  
   - **View Mode:**  
     - **Surfaces (f, fx, fy):** Renders the primary surface $f(x,y)$ and optionally $f_x$ and $f_y$.  
     - **Contour:** Displays a 2D contour map (often of “normals distance” or any scalar field).  
     - **Normals Distance:** Plots a 3D surface of the minimal distances between two normal lines.  
   - **Partial Surfaces:** Check the boxes to add $f_x$ or $f_y$ surfaces to your 3D visualization.  
   - **Grid Units:** The fineness of the discretization in $x$ and $y$. A higher value means more points but also higher computational load.  
   - **Z-Axis Limit:** Restricts the vertical axis range.

2. **Central Plot**  
   - Displays the chosen data as 3D surfaces, 2D contours, or scatter points (depending on mode) using Plotly or another library.

3. **Tools & Settings (Right Panel)**  
   - **Parameters (a, b, k):** Sliders to vary the offset $(a,b)$ for the second normal point, and a threshold or magnitude $k$.  
   - **Parameter Limits:** Further control the numeric range of these parameters.  
   - **Animations:**  
     - *Animate A, Animate B, Animate K*: Start or stop an animation that cycles these parameters, dynamically updating the plot.  

To use it effectively:
1. **Enter** your function in the “Function f(x,y)” box.  
2. **Enable** it and pick a **View Mode**.  
3. **Adjust** partial surface checkboxes if in Surfaces mode.  
4. **Set** $a$, $b$, $k$ with the sliders.  
5. **Optionally** animate $a$, $b$, or $k$.

---

## 5. Tips on Rendering Math and Markdown in a React App

- **Markdown Rendering:**  
  - [**react-markdown**](https://github.com/remarkjs/react-markdown) is a popular way to parse Markdown in React.  
  - [**markdown-it**](https://github.com/markdown-it/markdown-it) can also parse strings to HTML, which you can inject carefully.

- **Math Equations:**  
  - [**KaTeX**](https://katex.org/) or [**MathJax**](https://www.mathjax.org/) are standard for LaTeX in the browser.  
  - [**react-katex**](https://github.com/MatejBransky/react-katex) or [**react-mathjax**](https://github.com/z-green/react-mathjax) for friendly React integration.

- **Plotting Libraries:**  
  - [**react-plotly.js**](https://github.com/plotly/react-plotly.js) for interactive 2D and 3D charts.  
  - [**visx**](https://airbnb.io/visx/) or [**react-vis**](https://github.com/uber/react-vis) for lower-level custom charts.

Combining **react-markdown** for text and **react-katex** (or similar) for math fosters a robust, easy-to-maintain environment for your “Information Markdown” plus interactive plotting in React.
`;