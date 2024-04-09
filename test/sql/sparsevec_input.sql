SELECT '{1:1.5,3:3.5}/5'::sparsevec;
SELECT '{1:1.5,3:3.5}/5'::sparsevec::vector;

SELECT ''::sparsevec;
SELECT '{'::sparsevec;
SELECT '{}'::sparsevec;
SELECT '{}/'::sparsevec;
SELECT '{}/1a'::sparsevec;
SELECT '{:1}/1'::sparsevec;
SELECT '{1:}/1'::sparsevec;
SELECT '{1a:1}/1'::sparsevec;
SELECT '{1:1a}/1'::sparsevec;
SELECT '{1:1,}/1'::sparsevec;

SELECT '{1:0,2:1,3:0}/3'::sparsevec;

SELECT '{2:1,1:1}/2'::sparsevec;

SELECT '{}/5'::sparsevec;
SELECT '{}/-1'::sparsevec;
SELECT '{}/100001'::sparsevec;
SELECT '{}/16001'::sparsevec::vector;

SELECT '{0:1}/1'::sparsevec;
SELECT '{2:1}/1'::sparsevec;

SELECT '{}/1'::sparsevec(2);
SELECT '{1:1.5,3:3.5}/5'::sparsevec::vector(5);
SELECT '{1:1.5,3:3.5}/5'::sparsevec::vector(4);
