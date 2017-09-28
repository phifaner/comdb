#include <stdio.h>

int dtoa(double d, char *a)
  {
   int L = 0, S;
    double I = d;

    if (I < 1) 
    {
      S = 1; 
      L = S+1;
    }
    else 
    {
      // I: 
      while (I >= 1) 
      {
        L++;
        I /= 10;
      }

      S = L++;
    }

    int U;
    I = d - (int)(d);
    I *= 10;
    U = (int) (I);

    // printf("%lf, %lf\n", d, d - (int)(d));

    // error
    while (I > 0.001 && L < 8)
    {
      a[L] = '0' + U;

      // printf("a[%d]: %c\n", L, a[L]);
      if (a[L] >'9' || a[L] < '0') L++;
      else 
      {
        I = I - (int)(I);
        I *= 10;
        U = (int)(I);
        L++;
      }
    }

printf("-------L:%d, S:%d\n",  L, S);
    // in case of '120.0'
    if (L == S+1) 
    {
      a[L] = '0';
      L++;
    }

    // in case of integer
    if (L == 2) a[L++] = '0';
    a[L] = '\0';
    a[S--] = '.';

    I = d;
    // tackle LEFT part of '.'
    for (; I >= 10; --S)
    {
      a[S] = '0' + ((int)I)% 10;
      if (a[S] >'9' || a[S] < '0') --S;
      else I /= 10;
    }

    a[0] = '0' + (int)I;



        printf("-------%s, L:%d, S:%d\n", a, L, S);

    return L;
  }

  int itoa(int i, char *a)
  {
    int len = 0;
    int L = 0;
    int I = i;
    while (I > 0) 
    {
      L++;
      I /= 10;
    }

    len = L;
    a[L--] = '\0';

    for (; i >= 10; --L)
    {
      a[L] = '0' + i % 10;

      i /= 10;
    }

    a[L] = '0' + i;

                printf("%d, %s\n", len, a);
    return len;
  }


  int main()
  {
    char a[20];
    dtoa(130.0, a);

    // itoa(1, a);
  }