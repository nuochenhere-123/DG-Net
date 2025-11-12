
function dat_creat(score_matrix,feature_mat_scaled,fid,feature_vector_num,m,n)

for i = 1:m
    for j = 1:n
            query_str = [num2str(score_matrix((i-1)*n+j))...
                         ' qid:' num2str(i)];
            fprintf(fid, '%s', query_str);
        for k = 1:feature_vector_num
            
            query_str=[' ' num2str(k) ':' num2str(feature_mat_scaled((i-1)*n+j,k))];
            
            if k == feature_vector_num
              fprintf(fid, '%s\n', query_str);
            else
              fprintf(fid, '%s', query_str);  
            end
            
        end
            
    end
end

end
