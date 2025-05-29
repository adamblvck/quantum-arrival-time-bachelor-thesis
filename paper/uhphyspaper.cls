% Standard preamble for homework and tests
% Written by Allyson Robert

% Declares itself
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uhphyspaper}[2022/11/27 This class is for paper format reports written in LaTeX in the physics department at Hasselt university]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Packages         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\LoadClass[11pt, a4paper,twocolumn]{article}

%% Language package
\RequirePackage[english]{babel}

% Aesthetics, layout and font
\RequirePackage[margin={1in, 1in}, includeheadfoot, paperwidth=210mm, paperheight=297mm]{geometry}
\RequirePackage{titlesec} % to add a dot after section number
\RequirePackage{fancyhdr} % easy header and footer
% \RequirePackage{mathptmx} % Times New Roman font
\RequirePackage{titlecaps}% http://ctan.org/pkg/titlecaps
\RequirePackage{parskip} % blank space (single line) between paragraphs
\RequirePackage{setspace} % for setstretch command (line spacing)
% \RequirePackage[default,oldstyle,scale=0.95]{opensans} % Open Sans font is easier to read

% Good defaults to have
\RequirePackage{graphicx} % import images
\RequirePackage{authblk} % nultiple authors
\RequirePackage{float}

% Math and physics packages ... you will need them
\RequirePackage{amsmath}
\RequirePackage{physics}

% References and links (these should be last and in this order)
\RequirePackage[sorting=none,style=numeric]{biblatex}
\RequirePackage{xurl} % Allows links to break anywhere. See: https://tex.stackexchange.com/a/407368
\RequirePackage{csquotes}
\RequirePackage[colorlinks=true, linktoc=all]{hyperref}
\RequirePackage[noabbrev,dutch]{cleveref}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       New Elements       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand*{\course}[1]{\gdef\thecourse{#1}}
\newcommand*{\runningtitle}[1]{\gdef\therunningtitle{#1}}
\newcommand*{\bachyear}[1]{\gdef\thebachyear{#1}}
\newcommand*{\pimail}[1]{\gdef\thepimail{#1}}
\newcommand*{\pitel}[1]{\gdef\thepitel{#1}}
\newcommand*{\keywords}[1]{\gdef\thekeywords{#1}}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of the header %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\setlength{\headheight}{22.86868pt}
\rhead{\hspace*{1cm}\vspace*{0.05cm}\textit{\thecourse ~-~ \thebachyear Bachelor Physics}}
\lhead{\includegraphics[width=0.175\textwidth]{Figures/Logos/logo_uhasselt.png}}
\rfoot{\thepage}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Functional Definitions  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Resize the section titles
\titleformat{\section}
  {\normalfont\fontsize{11}{10}\bfseries}{\thesection}{1em}{}

% Puts a dot in the section number, avoids it in references
\makeatletter
\renewcommand{\@seccntformat}[1]{\csname the#1\endcsname.\quad}
\makeatother

% Edit spacing defaults
\titlespacing{\section}{0pt}{2ex}{0ex} % this one is important, especially last box
\titlespacing{\subsection}{0pt}{1ex}{0ex}
\titlespacing{\subsubsection}{0pt}{0.5ex}{0ex}

% column seperation distance
\setlength{\columnsep}{0.5cm} 

% Author management
\renewcommand\Affilfont{\fontsize{10}{10.8}\normalfont}
\renewcommand\Authfont{\fontsize{10}{10.8}\bfseries\selectfont}
\renewcommand*{\Authsep}{, }
\renewcommand*{\Authand}{, }
\renewcommand*{\Authands}{, }
\setlength{\affilsep}{2em}  
\newsavebox\affbox

% Shorthand for new paragraph
\newcommand{\npar}{\par\medskip} 

% Add the bibliography to the document
\addbibresource{references.bib}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Title Header       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\makeatletter
\def\@maketitle{%
    \begin{center}
        \textbf{\@title}
        \npar
        \textit{May 2025}
        \npar
        \@author
        \npar
    \end{center}
    
    \vspace{0.2cm}
    \textbf{Keywords:} \thekeywords\newline
	\small $^{\dagger}$\textit{Corresponding author.} 
	\textit{Tel: \color{cyan}{\thepitel}}; 
	\textit{E-mail address: \color{cyan}{\thepimail} }
	\npar
    \rule[1.5ex]{\linewidth}{0.4pt}
    \vspace{5mm}
}
\makeatother