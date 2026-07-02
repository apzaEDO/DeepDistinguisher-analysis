#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <stdint.h>

/* Paramètres nécessaires pour autom : n,m,t t appelé degg
    Sorties du programme : n,k,A et garder A si k = n-mt
    Elements du code C représenté comme des entiers unsigned int
    Python qui génére polynome et programme C doivent avoir meme polynome de corps et meme convention d'encodage des elements
    Reco gpt : fixer un polynome primitifs pour m=10.
    Format de sortie des matrices A : Format recommandé

    Un fichier batch temporaire avec :

    header :
    uint32 num_samples
    uint32 k
    uint32 n_minus_k
    puis les données uint8 aplaties
*/


/*  PROGRAMME QUI TOURNE 24/06/94 avec Optimisation matrice */
/* Version pour Florent Chabaud (Generation des matrices) */

/* PROGRAMME MODIFIE LE 26/04/95  UN FICHIE HH EST GENERE ET  */
/* CONTIENT LA MATRICE DE CONTROLE H SOUS LA FORME (I,M) */
/* LA MATRICE G GENEREE EST SOU LA FORME (Mt,I) */

/* version berlekamp avec deg controle pour les poly,matrice pointeur */
/* pas de reciproque */
/* euclide inclus , syndromes supplementaires calcules */
/* recherche des racines que dans L */
/* calcul des poids de la mat. generatrice */
/* option de suppression d'un sous-corps du corps de travail */

/* Les parametres sont bons si le poly utilise' n'a pas de racines doubles */
/* omega chosi de facon a ce que Tr(omega)=1 */

/* MODIFIE POUR CYTHERE NE GENERE PLUS QUE LES FICS POUR LA MAT GEN., LA MAT */
/* DE CONT. ET UN FICHIER DONNANT LA DIM DE LA MAT DE CONT */

#define Static static
#define maxdeg          400   /* degre des polynomes */
#define max1            (maxdeg << 1)   /* nombre de lignes max de H^2 */
#define maxsynd         (maxdeg << 1)   /* taille maximale d'un syndrome */
#define max2            4100   /* nombre de colonnes max de H et de G */


#define maxli           4000   /* nombre de lignes max de G */
#define max             4100   /* ~ nombre maximum d'elem du corps */
#define maxbase           30 /* nombre max de la base */

#define alf             'a'

#define hor             0   /* pour l'ecriture */
#define ver             1   /* des bits        */
#define nboptions       11
#define pr2             '='
#define pr1             '>'

typedef unsigned char boolean ;
typedef unsigned int element2[max + 2];   /* -1 pour 0 alpha[-1]=0 */
typedef unsigned int inverse[max + 1];
typedef int t_indice[max + 1];
typedef unsigned int tmot[max2];
typedef unsigned int syn[maxsynd];
typedef unsigned int polynome[maxdeg + 1];
/* typedef unsigned int mat_cont[max1 + 1][max2 + 1]; */
typedef unsigned int vecligne[max2];
typedef unsigned int *mat_gen[maxli];
typedef unsigned int *mat_cont[max1+1];
typedef char tmenu[nboptions][41];
typedef unsigned int tnom[maxli];


Static FILE *f;   /* Fichier des tables */
Static long dm;   /* 2^M longint pour M = 16 */
Static unsigned int alpha_m, n, dim;
/* alpha^M pour generer la table,
                                    longueur du code
                                    Dimension du code */
Static unsigned int m;
static int degg;
Static polynome g;   /* Polynome de GOPPA */
Static element2 alpha, valg;
/* Tableau des elements du corps,Valeurs inverses de G pour L */

Static mat_cont H;
Static inverse inv; 
Static mat_gen Gen;
Static t_indice indice;   /* pour afficher les matrices */
Static unsigned int L[max + 1];
Static char corps[7];
Static unsigned int posi, degg_shl_un;
Static FILE *fich;
Static unsigned int un_shl_mmoinsun;
Static unsigned int deu_degg_moinsun;
Static char pr0[9];
Static polynome lambda;
Static unsigned int taille_case;
static unsigned int maxcolG ; /* nombre d'octets necessaires pour stocker */
                              /* une ligne de G                           */
static unsigned int maxcase ;     /* nbre de colonnes maximum de G            */

static unsigned int capcorr,doublerac ;/* capacite de correction du code */
static unsigned int base[maxbase]; /* base du sous-corps */
static unsigned int newbase[maxbase]; /* nouvelle base du corps */
static unsigned int omega; /* element utilise pour la nouvelle base */
static unsigned int sousexpo ; /* exposant du sous-corps */

static unsigned int m_cible, t_cible,n_cible,seed;

/************* POUR LES TESTS ****************/

Static long cpt;
Static char f_NAME[6];
Static char fich_NAME[6];

static inline uint32_t low_mask_u32(unsigned int bits)
{
    if (bits == 0)  return 0u;
    if (bits >= 32) return 0xFFFFFFFFu;
    return (1u << bits) - 1u;
}

Static void mettre(unsigned int elem, unsigned int ligne, unsigned int colonne)
{
  unsigned int ccase, place;
  uint32_t m1, m2, masque1, masque2;

  ccase = colonne / taille_case;
  place = colonne % taille_case;
  if (place != 0)
    ccase++;
  if (place == 0) {
    place = taille_case;
    if (ccase == 0)
      ccase = 1;
  }

  m1 = (uint32_t)Gen[ligne - 1][ccase - 1];
  m2 = 0u;

  masque1 = low_mask_u32(place - 1);
  masque2 = m1 & low_mask_u32(taille_case - place);

  if (taille_case - place + 1 >= 32)
    m2 = 0u;
  else
    m2 = (m1 >> (taille_case - place + 1)) & masque1;

  m2 = (m2 << 1) | (elem & 1u);

  if (place != taille_case)
    m2 = (m2 << (taille_case - place)) | masque2;

  Gen[ligne - 1][ccase - 1] = m2;
}

Static unsigned int valeur (unsigned int ligne, unsigned int colonne)
{
  unsigned int ccase, place;

  ccase = colonne / taille_case;
  place = colonne % taille_case;
  if (place != 0)
    ccase++;
  if (place != 0)
    return ((Gen[ligne - 1][ccase - 1] >> (taille_case - place)) & 1);
  place = taille_case;
  if (ccase == 0)
    ccase = 1;
  return ((Gen[ligne - 1][ccase - 1] >> (taille_case - place)) & 1);
}


Static void changer_col(unsigned int i, unsigned int j)
{
  unsigned int elem1, elem2;
  unsigned int cpt;

  
  for (cpt = 1; cpt <= dim; cpt++) {
    elem1 = valeur(cpt, i);
    elem2 = valeur(cpt, j);
    mettre(elem1, cpt, j);
    mettre(elem2, cpt, i);
  }
}

Static void zero(unsigned int ligne, unsigned int colonne)
{
  unsigned int ccase, place;
  uint32_t tempo;

  ccase = colonne / taille_case;
  place = colonne % taille_case;
  if (place != 0)
    ccase++;
  if (place == 0) {
    place = taille_case;
    if (ccase == 0)
      ccase = 1;
  }

  tempo = 0u;
  if (place > 1) {
    tempo = low_mask_u32(place - 1);
    if (taille_case - place + 1 < 32)
      tempo <<= (taille_case - place + 1);
    else
      tempo = 0u;
  }

  Gen[ligne - 1][ccase - 1] &= tempo;
  memset(&Gen[ligne - 1][ccase], 0,
         (maxcase - ccase) * sizeof(Gen[0][0]));
}

Static void allouer(void)
{
  unsigned int j;

  for (j = 0; j < maxli; j++)
    Gen[j] = malloc(maxcase * sizeof(Gen[j][0]));
  for (j = 0; j < max1+1; j++) 
    H[j] = malloc(sizeof(int)*(max2+1));
/*puts("Allocation terminee\n");*/
}


Static void init(void)
{
  unsigned int j;

  memset(alpha, 0, sizeof(element2));
  alpha[0] = 0;
  memset(inv, 0, sizeof(inverse));
  memset(indice, 0, sizeof(t_indice));
  memset(g, 0, sizeof(polynome));
  memset(L, 0, sizeof(L));
  for (j=0; j < max1+1; j++)
    memset(H[j],0,sizeof(int)*(max2+1));
  for (j = 0; j < maxli; j++)
    memset(Gen[j], 0, maxcase*sizeof(Gen[j][0]));
  memset(valg, 0, sizeof(element2));
/*puts("Init ok\n");*/
}


Static void lecture(unsigned int *a)
{
  char car;
  unsigned int j, tempo, FORLIM;
  char STR1[256];

  *a = 0;
    for (j = 1; j <= m; j++) {
    car = getchar();
    sprintf(STR1, "%c", car);
    sscanf(STR1, "%d", &tempo);
    *a = ((*a) | tempo) << 1;
  }
  scanf("%*[^\n]");
  getchar();
  *a >>= 1;
}


Static void genere(unsigned int num)
{
  unsigned int j, tempo, reg;
  unsigned int FORLIM;

  alpha_m |= 1 << m;

  snprintf(f_NAME, sizeof(f_NAME), "F_2_%u", num);

  if (f != NULL)
    f = freopen(f_NAME, "w+b", f);
  else
    f = fopen(f_NAME, "w+b");

  if (f == NULL)
    puts("FileNotFound,genere");

  tempo = 1 << (num - 1);
  for (j = 0; j < num; j++)
    alpha[j + 1] = 1 << j;

  reg = tempo;
  FORLIM = dm;
  for (j = num + 1; j < FORLIM; j++) {
    if ((tempo & reg) != 0)
      reg = (reg << 1) ^ alpha_m;
    else
      reg <<= 1;
    alpha[j] = reg;
  }

  tempo = dm - 1;
  alpha[tempo + 1] = 1;
  fwrite(alpha, sizeof(alpha[1]), tempo + 1, f);

  FORLIM = dm - 2;
  for (j = 0; j <= FORLIM; j++) {
    inv[alpha[j + 1]] = alpha[tempo - j + 1];
    fwrite(&inv[alpha[j + 1]], sizeof(inv[1]), 1, f);
  }

  if (f != NULL)
    fclose(f);
  f = NULL;
}

Static void cree(void)
{
  int TEMP;

  printf("%s%c%c%c 0 pour aucune creation\n", pr0, pr2, pr2, pr1);
  printf("%s%c%c%cCreation de F_2^", pr0, pr2, pr2, pr1);
  scanf("%d%*[^\n]", &TEMP);
  getchar();
  m = TEMP;
  if (m == 0)
    return;
  dm = 2 << (m - 1);
  printf("%s%c%c%cEntrez %c^%d : ", pr0, pr2, pr2, pr1, alf, m);
  lecture(&alpha_m);
  genere(m);
  printf("%s%c%c%cCreation terminee ...\n", pr0, pr2, pr2, pr1);
  printf("beep\n");
}


Static void multiplie(unsigned int a, unsigned int b, unsigned int *res)
{
  /*a fixe , b bouge*/
  unsigned int i;

  *res = 0;
  for (i = 0; i < m; i++) {
    if ((a & (1 << i)) != 0)
      *res ^= b;
    if ((un_shl_mmoinsun & b) != 0)
      b = (b << 1) ^ alpha_m;
    else
      b <<= 1;
  }
}

Static unsigned int gg(unsigned int x)
{
  unsigned int aux;
  int TEMP;

  aux = g[degg];
  for (TEMP = degg - 1; TEMP >= 0; TEMP--) {
    multiplie(aux, x, &aux);
    aux ^= g[TEMP];
  }
  return aux;
}

Static void new_base()
{
  int j,test,tempo;

  srand(time(NULL));
  do test = rand() % dm ;
  while ((alpha[((test-1) << sousexpo) % (dm-1) + 1] == alpha[test]) || (gg(alpha[test])!=1)); 
  omega = alpha[test] ;  
  for ( j = 1; j <= m/2; j++) 
     newbase[j]=base[j];
  for ( j = m/2 +1; j <= m; j++)
     {
      multiplie(base[j-m/2],omega,&tempo);
      newbase[j]=tempo;
     }
   printf("Element omega choisi, tr(omega) : %c^%d %c^%d\n",alf,indice[omega],alf,indice[gg(omega)]);
   printf("Nouvelle base : \n");
   for ( j = 1; j <= m; j++)
   printf("%c^%d ",alf,indice[newbase[j]]);
   puts("\n");
}

Static void cherche_base(void)
{
  /* recherche une base d'un sous-corps */

  int basecor,j,TEMP,expo;

    printf("%s%c%c%cBase de F_2^", pr0, pr2, pr2, pr1);
    scanf("%d%*[^\n]", &TEMP);
    getchar();
    expo = TEMP;
    sousexpo=expo;
    if ((m % expo)==0) 
    {
     printf("Elements de F_2^%d \n",expo);
     for ( j = 0; j <= dm-2; j++ )
      if (alpha[((j << expo) + 1) % (dm - 1)] == alpha[j+1]) 
          {
            printf("%d ",j);
           };
      puts("\n");
      basecor=0;
      do
       {
        printf("Quels elements chosissez-vous pour la base ?\n");
        for ( j = 1; j <= expo; j++)
         {
           scanf("%d",&TEMP);
           base[j]=alpha[TEMP+1];
          }
          getchar();  /* lit le retour chariot qui reste */
         if (base[3]==(base[1] ^ base [2])) puts("Base incorrecte");
         else basecor=1;
        }
      while (basecor==0);
/*     for ( j = 1; j <= expo; j++)
      printf("%d ",indice[base[j]]); */
    }
}
	

Static void charger(unsigned int num)
{
  int j;
  int FORLIM;

  snprintf(corps, sizeof(corps), "%u", num);
  snprintf(f_NAME, sizeof(f_NAME), "F_2_%u", num);

  if (f != NULL)
    f = freopen(f_NAME, "r+b", f);
  else
    f = fopen(f_NAME, "r+b");

  if (f == NULL)
    puts("FileNotFound,charger");

  snprintf(corps, sizeof(corps), "F_2^%u", num);
  snprintf(pr0, sizeof(pr0), "(%s)", corps);

  fread(alpha, sizeof(alpha[1]), dm, f);
  FORLIM = dm;
  for (j = 1; j < FORLIM; j++)
    fread(&inv[alpha[j]], sizeof(inv[1]), 1, f);

  if (f != NULL)
    fclose(f);
  f = NULL;

  alpha_m = alpha[m + 1] | (1 << num);
  FORLIM = dm - 2;
  for (j = -1; j <= FORLIM; j++)
    indice[alpha[j + 1]] = j;

  un_shl_mmoinsun = 1 << (m - 1);
}

Static void charge(void)
{
  int TEMP;

  init();
  printf("%s%c%c%cchargement de F_2^", pr0, pr2, pr2, pr1);
  scanf("%d%*[^\n]", &TEMP);
  getchar();
  m = TEMP;
  dm = 2 << (m - 1);
  charger(m);
}


Static void analyse(void)
{
  FILE *fic;
  int puiss, coeff;
  char fic_NAME[12];

  fic = NULL;
  strcpy(fic_NAME, "tempo.txt");
  if (fic != NULL)
    fic = freopen(fic_NAME, "w", fic);
  else
    fic = fopen(fic_NAME, "w");
  if (fic == NULL)
    puts("FileNotFound,analyse");
  printf("%s%c%c%c< Syntaxe >: b_i i ,0=-1\n", pr0, pr2, pr2, pr1);
  printf("%s%c%c%cTaper -2 -2 pour terminer\n", pr0, pr2, pr2, pr1);
  printf("%s%c%c%cEntrez le polynome par puissances decroissantes : \n",
	 pr0, pr2, pr2, pr1);
  memset(g, 0, sizeof(polynome));
  scanf("%d%d%*[^\n]", &coeff, &puiss);
  getchar();
  degg = puiss;
  while (coeff != -2) {
    fprintf(fic, "%d %d ", coeff, puiss);
    g[puiss] = alpha[coeff+1];
    scanf("%d%d%*[^\n]", &coeff, &puiss);
    getchar();
  }
  if (fic != NULL)
    fclose(fic);
  fic = NULL;
  if (fic != NULL)
    fclose(fic);
}


Static void affiche_L(void)
{
  int j, indi;

  printf("%s%c%c%cL={ ", pr0, pr2, pr2, pr1);
  for (j = 0; j < n; j++) {
    indi = indice[L[j]];
    if (indi > 1)
      printf("%c^%d ", alf, indi);
    else {
      if (indi == 1)
	printf("%c ", alf);
      else {
	if (indi == 0)
	  printf("%d ", 1);
	else
	  printf("%d ", 0);
      }
    }
  }
  printf("}\n");
}


Static void mise_a_jour_L(void)
{
  int j, tempo, aux;
  char car;
  int loc[max + 1];
  unsigned int expo;
  int FORLIM;
  int TEMP;


  j = -1;
  do {
    printf(
      "%s%c%c%cConstruction de L : (M)anuelle,(A)leatoire,(S)uppression d'un sous-corps\n",
      pr0, pr2, pr2, pr1);
    printf("%s%c%c%c", pr0, pr2, pr2, pr1);
    scanf("%c%*[^\n]", &car);
    getchar();
  } while (toupper(car) != 'S' && toupper(car) != 'A' && toupper(car) != 'M');
  switch (toupper(car)) {

  case 'M':
    printf("%s%c%c%cRentrez les puissances de %c", pr0, pr2, pr2, pr1, alf);
    printf(" que vous desirez conserver (-1 pour 0) -2 pour terminer\n");
    printf("%s%c%c%c", pr0, pr2, pr2, pr1);
    aux=3;
    while (aux!=-2) {
      j++;
      scanf("%d", &aux);
      if (aux!=-2) L[j] = alpha[aux + 1];
    }
    n = j + 1;
    scanf("%*[^\n]");   /* on vide le buffer */
    getchar();
    break;

  case 'A':
    memset(loc, 0, sizeof(loc));
    memcpy(loc, L, n*sizeof(L[1]));
    aux = n;
    printf("%s%c%c%cCardinalite de L : ", pr0, pr2, pr2, pr1);
    scanf("%d%*[^\n]", &n);
    getchar();
    srand(rand() % 1821);
    memset(L, 0, sizeof(L));
    for (j = 0; j < n; j++) {
      tempo = rand() % aux;
      L[j] = loc[tempo];
      loc[tempo] = loc[aux - 1];
      aux--;
    }
    break;

  case 'S':
    aux = 0;
    printf("%s%c%c%cSuppression de F_2^", pr0, pr2, pr2, pr1);
    scanf("%d%*[^\n]", &TEMP);
    getchar();
    expo = TEMP;
    j = 0;
    while (aux != 1 << expo && j <= n - 2) {
      if (alpha[(indice[L[j]] << expo) % (dm - 1) + 1] == L[j]) {
	aux++;
	memcpy(&L[j], &L[j + 1], (n - j)*sizeof(L[1]));
	n--;
      } else
	j++;
    }
    if (alpha[(indice[L[n - 1]] << expo) + 1] == L[j])
      n--;
    break;

  }
  affiche_L();
}


Static void construit_L(void)
{
  int j, i, k;
  unsigned int val, inv_rac, aux, valcour;
  char car;
  int FORLIM;
  Static unsigned int valnul[max + 1];
  Static polynome poltempo;
 

  i = -1;
  k = -1 ;
  memset(poltempo,0,sizeof(poltempo));
  poltempo[0]=g[0];
  FORLIM = dm;
  for (j = 0; j < FORLIM; j++) {
    val = gg(alpha[j]);
    if (val != 0) {
      i++;
      L[i] = alpha[j];
      valg[L[i] + 1] = inv[val];
    }
  else
   {
    k++;
    valnul[k]=alpha[j];
   };
 };
n = i + 1;   /* longueur du code */
/* poltempo[degg-1]=g[degg];
doublerac=0;
  for (i = 0; i <= k; i++)
   {
    valcour=0;
    if (valnul[i]==0)
      {
        memcpy(poltempo, &g[1], degg*sizeof(int));
       }
    else  
       { 
        inv_rac=inv[valnul[i]];
        for (j = 0; j <= degg-2; j++)
         {
          multiplie(inv_rac, (valcour ^ g[j]), &aux);
          valcour=poltempo[j]=aux;
         }; 
        }; */
 /* Ici poltempo contient le polynome g(x) divise par (x-a_i) ou g(a_i)=0 */
  /* aux=poltempo[degg-1];
   for (j = degg-2; j >= 0; j--)
     {
      multiplie(aux, valnul[i], &aux);
      aux ^= poltempo[j];
     }
    if (aux==0) 
        {
          doublerac=1;
          i=k+1;
        };
   };
  if (doublerac==1) capcorr=degg/2 ;
  else 
      capcorr=degg ; 
  putchar('\n'); */
  capcorr=degg;
  /*affiche_L();*/
  /*
  do {
    printf(
      "%s%c%c%cDesirez-vous travailler avec (L) ou avec un (S)ous-ensemble de L\n",
      pr0, pr2, pr2, pr1);
    printf("%s%c%c%c", pr0, pr2, pr2, pr1);
    scanf("%c%*[^\n]", &car);
    getchar();
  } while (toupper(car) != 'S' && toupper(car) != 'L');
  if (toupper(car) == 'S')
    mise_a_jour_L();
  printf("%s%c%c%cConstruction du code,Patience ...\n", pr0, pr2, pr2, pr1);
  */
}


Static void diag_sup(void)
{
  unsigned int i, j, k, valaux;
  FILE *fic4;
  char fic4_NAME[13];


  for (i = dim; i >= 2; i--) { /*printf("Diagonalisation : %d\n",i);*/
    for (j = 1; j < i; j++) {
      if (valeur(j,i)!= 0) {
	for (k = dim+1; k <= n; k++){
          valaux = valeur(j, k) ^ valeur(i, k);
	  mettre(valaux, j, k);
	  }
      }
    }
  }

/*  ICI G CONTIENT EN FAIT LA MATRICE H DE LA FORME (I,M) */

/* ECRITURE DE LA MATRICE DE CONTROLE */
 
/*   fic4 = NULL;
   sprintf(fic4_NAME,"HH%d_%d_%d_%d",n,m*degg,dim,degg);
  if (fic4 != NULL)
    fic4 = freopen(fic4_NAME, "w", fic4);
  else
    fic4 = fopen(fic4_NAME, "w");
  if (fic4 == NULL)
    puts("FileNotFound,gen_fic");
  for (j = 1; j <= dim; j++) {
    for(i = 1; i <= j-1; i++) fprintf(fic4,"%d",0);
    fprintf(fic4,"%d",1);
    for(i = j+1; i <= dim; i++) fprintf(fic4,"%d",0);
    for (i = dim+1; i <= n; i++)
      fprintf(fic4, "%d", valeur(j,i));
    putc('\n', fic4);
  }
  if (fic4 != NULL)
    fclose(fic4); */

/* L'ETAPE CI DESSOUS CALCULE VRAIMENT G EN CALCULANT M transposee */

for (i = dim + 1; i <= n; i++) {
    k = i - dim;
    for (j = 1; j <= dim; j++){
       valaux=valeur(j,i);
       mettre(valaux,k,j);
      }
  }
  k = n - dim;
  for (i = 1; i <= k; i++) {   /* dim*/
    zero(i,dim+1);
    mettre(1,i,dim+i);
  }
}


static void swap_uint(unsigned int *a, unsigned int *b)
{
    unsigned int tmp = *a;
    *a = *b;
    *b = tmp;
}

Static void construit_G(void)
{
  unsigned int i, cpt, k, u, v, a, aux;
  int j;
  boolean test;
  void *p;
  unsigned int permut[max2];
  unsigned int colperm[max2];
  unsigned int newval, elem1, elem2;
  for (i = 0; i < n; i++) colperm[i] = i;
  unsigned int n_col_swaps = 0;
  memset(permut, 0, sizeof(permut));
  dim = degg * m;
  if (dim > n) {
    dim = 0;
    return;
  }
   for (j = 0; j < n; j++) {
    for (i = 0; i < degg; i++) {
      a = i * m;
      for (cpt = 0; cpt < m; cpt++){
     	newval = (H[i][j] >> (m - cpt - 1)) & 1;
	mettre(newval, a + cpt + 1, j + 1);  
     }
    }
  }
/* puts("passe"); */
  i = 1;
  while (i <= dim) { /*printf("Triangularisation : %d  \n",i);*/
    j = i;
    /* parcours de la colonne pivot */
    while ((j <= dim) && valeur(j,i) != 1)
      j++;
    if (j > dim) {
      k = i + 1;
      /* parcours de la ligne pivot */
      while ((k <= n) && valeur(i,k) != 1)
	k++;
      if (k > n) {
      /* parcours de la zone rectangulaire SE */
	v = i + 1;
	test = (i != dim);
	while ((test) && (v <= n)) {
	  u = i + 1;
	  while ((test) && (u <= dim)) {
	    test = ((valeur(u,v) != 1) && (test));
	    if (test)
	      u++;
	  }
	  if (test)
	    v++;
	}
	if (test || i == dim) {
	/* la matrice n'est pas de dimension max */
	  dim = i - 1;
	  i = dim;
	} else {
	  p = Gen[i - 1];
	  Gen[i - 1] = Gen[u - 1];
	  Gen[u - 1] = p;
          changer_col(i,v); 
	  permut[i - 1] = v;
    n_col_swaps++;
    swap_uint(&colperm[i - 1], &colperm[v - 1]);
	}
      } else {
      /* le pivot est dans la ligne pivot */
        changer_col(i,k);
	permut[i - 1] = k;
  n_col_swaps++;
  swap_uint(&colperm[i - 1], &colperm[k - 1]);
      }
    } else {
    /* le pivot est dans la colonne pivot */
      if (j != i) {
	p = Gen[i - 1];
	Gen[i - 1] = Gen[j - 1];
	Gen[j - 1] = p;
      }
    }
    for (j = i+1; j <= dim; j++) {
      if (valeur(j,i) != 0) {
	for (k = i+1; k <= n; k++){
	  newval = valeur(j, k) ^ valeur(i, k);
	  mettre(newval, j, k);
       }
      }
    }
    i++;
  }
 /*  puts("salut"); */
  diag_sup();
 /* for (j = n; j >= 1; j--) {
    aux = permut[j-1];
    if (aux != 0) {
      for (k = 1; k <= n-dim; k++) {  
	elem1 = valeur(k, j);
	elem2 = valeur(k, aux);
	mettre(elem2, k, j);
	mettre(elem1, k, aux);
      }
    }
  } */
  dim = n - dim;
  /*for (i=0;i<max2; i++){
    printf("%i\n",colperm[i]);
  }*/
 /*
  fprintf(stderr, "n_col_swaps = %u\n", n_col_swaps);
  fprintf(stderr, "colperm:");
  for (i = 0; i < n; i++) fprintf(stderr, " %u", colperm[i]);
  fprintf(stderr, "\n");*/
}

Static void construit_G_rejet_col(void)
{
  unsigned int i, cpt, k, a, newval;
  int j;
  void *p;

  dim = degg * m;
  if (dim > n) {
    dim = 0;
    return;
  }

  /* Remplissage binaire de H dans Gen */
  for (j = 0; j < n; j++) {
    for (i = 0; i < degg; i++) {
      a = i * m;
      for (cpt = 0; cpt < m; cpt++) {
        newval = (H[i][j] >> (m - cpt - 1)) & 1;
        mettre(newval, a + cpt + 1, j + 1);
      }
    }
  }

  /* Triangularisation SANS échanges de colonnes :
     on autorise seulement des échanges de lignes.
     Si la colonne i n'a pas de pivot dans les lignes i..dim,
     on rejette l'échantillon. */
  i = 1;
  while (i <= dim) {
    j = i;

    /* chercher un pivot dans la colonne i */
    while ((j <= dim) && valeur(j, i) != 1)
      j++;

    /* pas de pivot dans la colonne courante -> non standard, on jette */
    if (j > dim) {
      dim = 0;
      return;
    }

    /* échange de lignes seulement */
    if (j != i) {
      p = Gen[i - 1];
      Gen[i - 1] = Gen[j - 1];
      Gen[j - 1] = p;
    }

    /* élimination sous le pivot */
    for (j = i + 1; j <= dim; j++) {
      if (valeur(j, i) != 0) {
        for (k = i + 1; k <= n; k++) {
          newval = valeur(j, k) ^ valeur(i, k);
          mettre(newval, j, k);
        }
      }
    }

    i++;
  }

  diag_sup();

  /* à la fin, dim devient la dimension k du générateur G=(A|I) */
  dim = n - dim;
}


Static void construit_H(void)
{
  unsigned int i, j, tempo, aux;


  for (j = 0; j < n; j++)
    H[0][j] = valg[L[j] + 1];
  for (j = 0; j < n; j++) {
   
    tempo = H[0][j];
    aux = L[j];
      for (i = 1; i < degg; i++) {
      multiplie(tempo, aux, &tempo);
      H[i][j] = tempo;
    }
  }
/* for (j = 0; j < n; j++)
   {
    multiplie(H[degg-1][j],L[j],&tempo);
    H[1][j]=tempo;
   } */
}


Static void construit_H2(void)
{
  unsigned int i, j, tempo, aux;

  for(j = 0; j < max1+1; j++)
     memset(H[j],0,sizeof(int)*(max2+1));
  for (j = 0; j < n; j++)
    multiplie(valg[L[j] + 1], valg[L[j] + 1], &H[0][j]);
  for (j = 0; j < n; j++) {
    tempo = H[0][j];
    aux = L[j];
    for (i = 1; i <= deu_degg_moinsun ; i++) {
      multiplie(tempo, aux, &tempo);
      H[i][j] = tempo;
    }
  }
}


Static void lire_poly(void)
{
  unsigned int j;

  memset(g, 0, sizeof(polynome));
  memset(L, 0, sizeof(L));
  for (j = 0; j < max1+1; j++)
     memset(H[j],0,sizeof(int)*(max2+1));
  for (j = 0; j < maxli; j++)
    memset(Gen[j], 0, maxcase*sizeof(Gen[j][0]));
  memset(valg, 0, sizeof(element2));
  analyse();
  deu_degg_moinsun = (degg << 1) - 1;
  degg_shl_un = degg << 1;
  construit_L();
  construit_H(); /* puts("H"); */
  construit_G(); /* puts("G"); */
  construit_H2();  /* puts("H2"); */
}


Static void calcul_syndrome(unsigned int *a, unsigned int *s)
{
  unsigned int i, j, som;

  memset(s, 0, sizeof(syn));
  for (i = 0; i <=deu_degg_moinsun ; i++) {
    som = 0;
    for (j = 0; j < n; j++)
      som ^= a[j] * H[i][j];
    s[i] = som;
  }
}


Static void ecritmot(unsigned int *a)
{
  unsigned int j, FORLIM;

  FORLIM = n;
  for (j = 0; j < FORLIM; j++)
    printf("%d", a[j]);
}


Static unsigned int lambd(unsigned int x)
{
  unsigned int aux;
  unsigned int j, FORLIM;

  aux = 1;   /*lambda[0]*/
  FORLIM = degg;
  for (j = 1; j <= FORLIM; j++) {
    multiplie(aux, x, &aux);
    aux ^= lambda[j];
  }
  return aux;
}


Static void correction(unsigned int *corrige)
{
  unsigned int zeros, val, i;

  zeros = 0;
  i = 0;
  while ((i < n) && (zeros < degg)) {
    val = lambd(L[i]);
    if (val == 0) {
      zeros++;
      corrige[i] ^= 1;
    }
    i++;
  }
}


Static void decode_berley(unsigned int *s, unsigned int *corrige)
{
  unsigned int r, long_, j, tempo, delta, taille, deglambda, degb, degt;
  polynome b, u, zero;



  taille = sizeof(polynome);
  memset(lambda, 0, taille);
  memset(b, 0, taille);
  r = 0;
  long_ = 0;
  lambda[0] = 1;
  b[0] = 1;
  deglambda = 0;
  degb = 0;
  tempo = 0;
  do 
  {
    r++;
    delta = s[r - 1];
    for (j = 1; j <= long_; j++) 
      {
        multiplie(lambda[j], s[r - j - 1], &tempo);
        delta ^= tempo;
      }
    if (delta != 0) 
      {
        memset(u, 0, taille);
        u[0] = 1;   /*lambda[0]*/
        if (deglambda > degb + 1)
	       degt = deglambda;
        else
	       degt = degb + 1;
        for (j = 1; j <= degt; j++) 
          {   /* maxdeg */
	           multiplie(delta, b[j - 1], &tempo);
	           u[j] = lambda[j] ^ tempo;
          }
        while (u[degt] == 0 && degt > 0)
	         degt--;
        if (long_ << 1 >= r) 
          {
	         memset(zero, 0, taille);
	         memcpy(&zero[1], b, (degb + 1)*sizeof(b[1]));
	         memset(b, 0, taille);
	         degb++;
	         memcpy(b, zero, (degb + 1)*sizeof(b[1]));
          } 
          else 
          {
	         tempo = inv[delta];
	         degb = deglambda;
	         memset(b, 0, taille);
	         for (j = 0; j <= degb; j++)
	           multiplie(lambda[j], tempo, &b[j]);
	         long_ = r - long_;
          }
        deglambda = degt;
        memset(lambda, 0, taille);
        memcpy(lambda, u, taille - sizeof(b[1]));
      } 
      else 
      {
        memset(zero, 0, taille);
        memcpy(&zero[1], b, (degb + 1)*sizeof(b[1]));
        memset(b, 0, taille);
        degb++;
        memcpy(b, zero, (degb + 1)*sizeof(b[1]));
      }
  } 
  while (r < degg_shl_un);
  correction(corrige);
}


Static void ecritsyndrome(unsigned int *s)
{
  unsigned int j;

  printf("SYNDROME : \n");
  for (j = 0; j < degg; j++)
    printf("%u ", s[j]);
}


Static unsigned int poids(unsigned int *mot, unsigned int longueur)
{
  unsigned int cpt, somme, j;

  somme = 0;
  for (cpt = 0; cpt < longueur; cpt++)
  {
    for (j=0; j < taille_case; j++)
    somme += (mot[cpt] >> j) & 1;
   }
  return somme;
}

Static void affiche_G_interactif(void)
{
  unsigned int sb_inf, sb_sup, ab_inf, ab_sup, a, s, i, j, k, l, newval;
  char carlu;

  carlu = ' ';
  printf("%s%c%c%cMatrice generatrice du code : \n", pr0, pr2, pr2, pr1);
  a = n >> 4;
  if ((n & 15) != 0)
    a++;
  s = dim >> 4;
  if ((dim & 15) != 0)
    s++;
  do {
    if (carlu != '*') {
      for (k = 1; k <= a; k++) {
	if (carlu != '*') {
	  ab_inf = ((k - 1) << 4) + 1;
	  ab_sup = k << 4;
	  if (ab_sup > n)
	    ab_sup = n;
	  for (l = 1; l <= s; l++) {
	    if (carlu != '*') {
	      printf("%7u", ab_inf);
	      for (j = ab_inf + 1; j <= ab_sup; j++)
		printf("%4u", j);
	      printf("\n\n");
	      sb_inf = ((l - 1) << 4) + 1;
	      sb_sup = l << 4;
	      if (sb_sup > dim)
		sb_sup = dim;
	      for (i = sb_inf; i <= sb_sup; i++) {
		printf("%3u", i);
		for (j = ab_inf ; j <= ab_sup; j++){
		  newval = valeur(i, j);
		  printf("%4d", newval);
               }
		putchar('\n');
	      }
              putchar('\n');
	      if (dim > 16 || n>16)
		printf("%s%c%c%cAppuyez sur une touche - Sortie : *\n",
		       pr0, pr2, pr2, pr1);
	      else
		printf("%s%c%c%cSortie : *\n", pr0, pr2, pr2, pr1);
	      printf("%s%c%c%c", pr0, pr2, pr2, pr1);
	      scanf("%c%*[^\n]", &carlu);
	      getchar();
	    }
	  }
	}
      }
    }
  } while (carlu != '*');
}


Static void affiche_G(void)
{
  unsigned int i, j, newval;

  printf("%s%c%c%cMatrice generatrice du code : \n", pr0, pr2, pr2, pr1);

  /* entête colonnes */
  printf("%7u", 1);
  for (j = 2; j <= n; j++)
    printf("%4u", j);
  printf("\n\n");

  /* lignes de G */
  for (i = 1; i <= dim; i++) {
    printf("%3u", i);
    for (j = 1; j <= n-dim; j++) {
      newval = valeur(i, j);
      printf("%4u", newval);
    }
    putchar('\n');
  }
  putchar('\n');
}

Static void affiche_param(void)
{
  FILE *fic;
  unsigned int i, j;
  int puiss, coeff, tempo;
  unsigned int n1,FORLIM;
  char fic_NAME[12];
  int test ;

  fic = NULL;
  n1=n / taille_case;
  if ((n % taille_case)!=0) n1++;
  strcpy(fic_NAME, "tempo.txt");
  if (fic != NULL)
    fic = freopen(fic_NAME, "r", fic);
  else
    fic = fopen(fic_NAME, "r");
  if (fic == NULL)
    puts("FileNotFound,affiche_param,1");
  printf("%s%c%c%cLongueur               : %u\n", pr0, pr2, pr2, pr1, n);
  printf("%s%c%c%cDimension              : %u\n", pr0, pr2, pr2, pr1, dim);
  if (doublerac) tempo=degg+1;
       else tempo = (degg << 1) + 1;
  printf("%s%c%c%cDistance minimum       : >=%d\n", pr0, pr2, pr2, pr1, tempo);
  printf("%s%c%c%cCapacite de correction : >=%d\n", pr0, pr2, pr2, pr1,capcorr);
  i = 0;
  j = 0;
  printf("%s%c%c%cPolynome               : ", pr0, pr2, pr2, pr1);
  test=fscanf(fic, "%d", &coeff);
  while (test!=EOF) {
    test=fscanf(fic, "%d", &puiss);
    if (coeff == -1 && puiss == 0)
      printf("%d", 0);
    else {
      if (coeff == 0 && puiss == 0)
	printf("%d", 1);
      else {
	if (coeff != -1 && coeff != 0) {
	  printf("%c^%d", alf, coeff);
	  if (puiss != 0)
	    putchar('*');
	}
      }
    }
    if (puiss != 0 && puiss != 1 && coeff !=-1)
      printf("x^%d", puiss);
    else {
      if (puiss == 1 && coeff !=-1)
	putchar('x');
    }
    test=fscanf(fic, "%d", &coeff);
    if (test!=EOF) 
      putchar('+');
  }
  if (fic != NULL)
    fclose(fic);
  fic = NULL;
  printf("\n%s%c%c%cAppuyez sur return ", pr0, pr2, pr2, pr1);
  scanf("%*[^\n]");
  getchar();
  affiche_L();
  printf("%s%c%c%cPoids de la matrice generatrice crees : \n\n",
	 pr0, pr2, pr2, pr1);
  strcpy(fic_NAME, "poids.txt");
  if (fic != NULL)
    fic = freopen(fic_NAME, "w", fic);
  else
    fic = fopen(fic_NAME, "w");
  if (fic == NULL)
    puts("FileNotFound,affiche_param,2");
  FORLIM = dim;
  for (j = 1; j <= FORLIM; j++) {
    tempo = poids(Gen[j - 1], n1);
    fprintf(fic, "%4d", tempo);
    if (j % 20 == 0)
      putc('\n', fic);
  }
  putc('\n',fic);
  if (fic != NULL)
    fclose(fic);
  fic = NULL;
  if (fic != NULL)
    fclose(fic);
}


Static void gennom(unsigned int *u)
{
  unsigned int i;

  srand(time(NULL));
  memset(u, 0, sizeof(tnom));
  for (i = 0; i < dim; i++)
    u[i] = rand() & 1;
}


Static void codemot(unsigned int *u, unsigned int *mot)
{
  unsigned int i, j;
  unsigned int somme;
 
  for (i = 1; i <= n; i++) {
    somme = 0;
    for (j = 1; j <= dim; j++)
      somme ^= valeur(j,i) & u[j-1];
    mot[i-1] = somme;
  }
}


Static void gen_erreur(unsigned int *mot, unsigned int *err)
{
  unsigned int i, aux, tempo;
  unsigned int loc[max2];
  unsigned int FORLIM;

  memset(loc, 0, sizeof(loc));
  memset(err, 0, sizeof(tmot));
  srand(rand() % time(NULL));
  aux = n;
  memcpy(err, mot, n*sizeof(mot[1]));
  for (i = 1; i <= n; i++)
    loc[i - 1] = i;
  for (i = 1; i <= degg; i++) {
    tempo = (rand() % aux) + 1;
    err[loc[tempo - 1] - 1] = mot[loc[tempo - 1] - 1] ^ 1;
    loc[tempo - 1] = loc[aux - 1];
    aux--;
  }
}


Static void verif(unsigned int *mot)
{
  unsigned int i, j, som, FORLIM, FORLIM1;


  for (i = 0; i < degg; i++) {
    som = 0;
    for (j = 0; j < n; j++)
      som ^= H[i][j] * mot[j];
  }
}


Static void decode(void)
{
  unsigned int j;
  tmot err, mot;
  syn s;
  tnom nom;

  memset(err, 0, sizeof(tmot));
  memset(s, 0, sizeof(syn));
  memset(mot, 0, sizeof(tmot));
  gennom(nom);
  printf("%s%c%c%cMot du code : \n", pr0, pr2, pr2, pr1);
  codemot(nom, mot);
  ecritmot(mot);
  putchar('\n');
  gen_erreur(mot, err);
  printf("%s%c%c%cMot erreur : \n", pr0, pr2, pr2, pr1);
  ecritmot(err);
  putchar('\n');
  calcul_syndrome(err, s);
  decode_berley(s, err);
  printf("%s%c%c%cCorrection : \n", pr0, pr2, pr2, pr1);
  for (j = 0; j < dim; j++)
    printf("%d", err[j]);
  //putchar('*');
  printf("%d", err[dim]);
  //putchar('*');
  for (j = dim + 1; j < n; j++)
    printf("%d", err[j]);
  putchar('\n');
  putchar('\n');
}


/* Static void gen_fic(void)
{
  FILE *fic2;
  FILE *fic1, *fic3;
  unsigned int j, FORLIM;
  char fic2_NAME[11];
  char fic1_NAME[11];
  char fic3_NAME[11];

  fic3 = NULL;
  fic1 = NULL;
  fic2 = NULL;
  strcpy(fic1_NAME, "gen.txt");
  if (fic1 != NULL)
    fic1 = freopen(fic1_NAME, "w+b", fic1);
  else
    fic1 = fopen(fic1_NAME, "w+b");
  if (fic1 == NULL)
    puts("FileNotFound,gen_fic");
  for (j = 0; j < dim; j++)
    fwrite(Gen[j], sizeof(Gen[1][1]), n, fic1);
  if (fic1 != NULL)
    fclose(fic1);
  fic1 = NULL;
  strcpy(fic2_NAME, "cont.txt");
  if (fic2 != NULL)
    fic2 = freopen(fic2_NAME, "w+b", fic2);
  else
    fic2 = fopen(fic2_NAME, "w+b");
  if (fic2 == NULL)
    puts("FileNotFound,gen_fic");
  fwrite(H, sizeof(mat_cont), 1, fic2);
  if (fic2 != NULL)
    fclose(fic2);
  fic2 = NULL;
  strcpy(fic3_NAME, "L.txt");
  if (fic3 != NULL)
    fic3 = freopen(fic3_NAME, "w+b", fic3);
  else
    fic3 = fopen(fic3_NAME, "w+b");
  if (fic3 == NULL)
    puts("FileNotFound,gen_fic");
  fwrite(&dim, sizeof(dim), 1, fic3);
  fwrite(&n, sizeof(n), 1, fic3);
  fwrite(&degg, sizeof(degg), 1, fic3);
  fwrite(L, sizeof(L[1]), n, fic3);
  if (fic3 != NULL)
    fclose(fic3);
  fic3 = NULL;
  printf("%s%c%c%cFichiers generes\n", pr0, pr2, pr2, pr1);
  if (fic2 != NULL)
    fclose(fic2);
  if (fic1 != NULL)
    fclose(fic1);
  if (fic3 != NULL)
    fclose(fic3);
}
*/

Static void gen_fic_A(int idx)
{
  FILE *fic;
  unsigned int i, j;
  char name[64];

  sprintf(name, "A_%04d_%u_%u_%d.txt", idx, n, dim, degg);
  fic = fopen(name, "w");
  if (fic == NULL) {
    puts("FileNotFound,gen_fic_A");
    return;
  }

  /* G est stockée sous la forme (A | I_dim), donc on écrit seulement A */
  for (j = 1; j <= dim; j++) {
    for (i = 1; i <= n - dim; i++) {
      fprintf(fic, "%d", valeur(j, i));
    }
    putc('\n', fic);
  }

  fclose(fic);
}


Static void quitter(void)
{
  unsigned int j;

  for (j = 0; j < maxli; j++)
    free(Gen[j]);
  for (j = 0; j< max1+1; j++)
    free(H[j]);
 }

void setup_field(unsigned int m_in, unsigned int alpha_m_in)
{
    int j;

    m = m_in;
    dm = 2 << (m - 1);
    alpha_m = alpha_m_in;

    genere(m);

    /* ce que charger(m) t'apporte vraiment */
    for (j = -1; j <= (int)dm - 2; j++)
        indice[alpha[j + 1]] = j;

    un_shl_mmoinsun = 1 << (m - 1);
}

void reset_code_state(void)
{
    unsigned int j;

    memset(g, 0, sizeof(polynome));
    memset(L, 0, sizeof(L));
    memset(valg, 0, sizeof(element2));

    for (j = 0; j < max1 + 1; j++)
        memset(H[j], 0, sizeof(H[j][0]) * (max2 + 1));

    for (j = 0; j < maxli; j++)
        memset(Gen[j], 0, maxcase * sizeof(Gen[j][0]));
}

static unsigned int field_alpha_m_lowbits(unsigned int m)
{
    switch (m) {
        case 3:
            /* x^3 + x + 1  => alpha^3 = alpha + 1 */
            return 3;   /* 0b011 */

        case 10:
            /* exemple: x^10 + x^3 + 1 => alpha^10 = alpha^3 + 1 */
            return 9;   /* 0b0000001001 */

        default:
            fprintf(stderr, "Ajoute un polynome de corps pour m=%u\n", m);
            exit(1);
    }
}

static int read_goppa_poly_stdin(void)
{
    unsigned int i, c;

    memset(g, 0, sizeof(polynome));
    degg = (int)t_cible;

    for (i = 0; i < t_cible; i++) {
        if (scanf("%u", &c) != 1)
            return 0;   /* EOF */
        g[i] = c;
    }

    g[t_cible] = 1;   /* polynôme monique */

    deu_degg_moinsun = (degg << 1) - 1;
    degg_shl_un = degg << 1;

    return 1;
}

static void randomize_support(unsigned int target_n)
{
    unsigned int i, j, tmp;
    unsigned int full_n = n;

    if (target_n == 0 || target_n > full_n) {
        fprintf(stderr, "target_n invalide: %u (support dispo: %u)\n", target_n, full_n);
        n = 0;
        return;
    }

    for (i = full_n - 1; i > 0; i--) {
        j = (unsigned int)(rand() % (i + 1));
        tmp = L[i];
        L[i] = L[j];
        L[j] = tmp;
    }

    n = target_n;
}

static void fisher_yates_uint(unsigned int *perm, unsigned int len)
{
    unsigned int i, j;
    if (len <= 1) return;

    for (i = len - 1; i > 0; i--) {
        j = (unsigned int)(rand() % (i + 1));
        swap_uint(&perm[i], &perm[j]);
    }
}

static void shuffle_dense_A(unsigned char *A, unsigned int k, unsigned int nA)
{
    unsigned int i, j;
    unsigned int *prow = NULL;
    unsigned int *pcol = NULL;
    unsigned char *B = NULL;

    prow = (unsigned int *)malloc(k * sizeof(unsigned int));
    pcol = (unsigned int *)malloc(nA * sizeof(unsigned int));
    B    = (unsigned char *)malloc(k * nA * sizeof(unsigned char));

    if (prow == NULL || pcol == NULL || B == NULL) {
        fprintf(stderr, "shuffle_dense_A: malloc failed\n");
        free(prow);
        free(pcol);
        free(B);
        exit(1);
    }

    for (i = 0; i < k; i++) {
        prow[i] = i;
    }
    for (j = 0; j < nA; j++) {
        pcol[j] = j;
    }

    /* permutation aléatoire des lignes et colonnes de A */
    fisher_yates_uint(prow, k);
    fisher_yates_uint(pcol, nA);

    for (i = 0; i < k; i++) {
        for (j = 0; j < nA; j++) {
            B[i * nA + j] = A[prow[i] * nA + pcol[j]];
        }
    }

    memcpy(A, B, k * nA * sizeof(unsigned char));

    free(prow);
    free(pcol);
    free(B);
}

static unsigned char *export_A_dense(void)
{
    unsigned int i, j, nA;
    unsigned char *buf;

    nA = n - dim;   /* largeur de A dans G = (A | I) */
    buf = malloc(dim * nA * sizeof(unsigned char));
    if (buf == NULL)
        return NULL;

    for (i = 1; i <= dim; i++) {
        for (j = 1; j <= nA; j++) {
            buf[(i - 1) * nA + (j - 1)] = (unsigned char) valeur(i, j);
        }
    }

    return buf;
}

int genere_A(void)
{
    uint32_t k, nA;
    unsigned char *Aout;
    unsigned int full_n;
    unsigned int expected_k;

    init();

    m = m_cible;
    dm = 2 << (m - 1);
    /*alpha_m = field_alpha_m_lowbits(m);*/

    setup_field(m, alpha_m);

    while (1) {
        reset_code_state();

        if (!read_goppa_poly_stdin())
            break;   /* EOF */

        /* support complet des non-racines */
        construit_L();
        full_n = n;

        if (n_cible > full_n) {
            fprintf(stderr, "n_target=%u > support dispo=%u\n", n_cible, full_n);
            continue;
        }

        if (n_cible < full_n) {
            randomize_support(n_cible);
        } else {
            n = full_n;
        }

        construit_H();
        construit_G_rejet_col();

        expected_k = n_cible - m_cible * t_cible;
        if (dim != expected_k) {
            /* on jette l'échantillon si le rang n'est pas celui attendu */
            continue;
        }

        Aout = export_A_dense();
        if (Aout == NULL) {
            fprintf(stderr, "malloc failed\n");
            return 0;
        }

        k = (uint32_t) dim;
        nA = (uint32_t) (n - dim);

        /* important : casse le biais positionnel introduit par la systématisation */
        /*shuffle_dense_A(Aout, k, nA);*/

        fwrite(&k, sizeof(uint32_t), 1, stdout);
        fwrite(&nA, sizeof(uint32_t), 1, stdout);
        fwrite(Aout, sizeof(unsigned char), k * nA, stdout);
        free(Aout);
    }

    return 1;
}


int main(int argc, char *argv[])
{
    fich = NULL;
    f = NULL;

    if (argc != 6) {
        fprintf(stderr, "Usage: %s m t n alpha_m seed\n", argv[0]);
        return 1;
    }

    m_cible = (unsigned int)atoi(argv[1]);
    t_cible = (unsigned int)atoi(argv[2]);
    n_cible = (unsigned int)atoi(argv[3]);
    alpha_m = (unsigned int)atoi(argv[4]);
    seed = (unsigned int)atoi(argv[5]);
    taille_case = sizeof(int) * 8;
    maxcolG = (max2 >> 3);
    if ((max2 & 7) != 0) maxcolG++;
    maxcase = maxcolG / sizeof(int);
    if ((maxcolG % sizeof(int)) != 0) maxcase++;

    *pr0 = '\0';
    *corps = '\0';

    srand(seed);

    allouer();
    genere_A();
    quitter();
    return 0;
}


/* End. */
